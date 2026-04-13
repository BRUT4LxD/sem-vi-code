
import os
import csv
import datetime
import traceback
import torch
from typing import List, Dict, Tuple, Optional, Set
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np

from attacks.attack_factory import AttackFactory
from attacks.attack_names import AttackNames
from data_eng.io import load_model_imagenette
from data_eng.dataset_loader import load_imagenette
from domain.model.model_names import ModelNames
from shared.model_utils import ModelUtils


def _resolve_saved_adv_images_dir(
    attacked_images_folder: str, source_model_name: str, attack_name: str
) -> Optional[str]:
    """
    Resolve directory with saved adversarial PNGs.

    Supports layout from imagenette_adv_imgs_generator:
        {root}/{train|test}/{model}/{attack}/{class}/*.png
    and legacy flat layout:
        {root}/{model}/{attack}/{class}/*.png
    """
    candidates = [
        os.path.join(attacked_images_folder, "train", source_model_name, attack_name),
        os.path.join(attacked_images_folder, "test", source_model_name, attack_name),
        os.path.join(attacked_images_folder, source_model_name, attack_name),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


# Short CSV basename tags for incremental / bulk CSV (filename_key, method) -> prefix
_TRANSFERABILITY_CSV_PREFIX: Dict[Tuple[str, str], str] = {
    ("model2model_transferability", "in_memory"): "m2m_trans_mem",
    ("attack2model_transferability", "in_memory"): "a2m_trans_mem",
    ("model2model_transferability", "from_files"): "m2m_trans_files",
    ("attack2model_transferability", "from_files"): "a2m_trans_files",
}

TRANSFERABILITY_CSV_FIELDNAMES: List[str] = [
    "source_model",
    "target_model",
    "attack_name",
    "total_images",
    "total_successful_attacks",
    "transfer_success",
    "transfer_rate",
    "attack_success_rate",
    "timestamp",
]


def _transferability_csv_path(results_folder: str, filename: str, method: str) -> str:
    prefix = _TRANSFERABILITY_CSV_PREFIX.get((filename, method))
    if prefix is None:
        prefix = f"{filename}_{method}".replace(" ", "_")
    return os.path.join(results_folder, f"{prefix}.csv")


def _load_recorded_transfer_keys(csv_path: str) -> Set[Tuple[str, str, str]]:
    """
    Keys already present in the incremental CSV: (source_model, target_model, attack_name).
    """
    recorded: Set[Tuple[str, str, str]] = set()
    if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
        return recorded
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return recorded
            for row in reader:
                try:
                    s = row.get("source_model", "").strip()
                    t = row.get("target_model", "").strip()
                    a = row.get("attack_name", "").strip()
                    if s and t and a:
                        recorded.add((s, t, a))
                except (TypeError, AttributeError):
                    continue
    except OSError:
        pass
    return recorded


class TransferabilityResult:
    """Class to store transferability attack results"""
    
    def __init__(self, source_model: str, target_model: str, attack_name: str, 
                 transfer_success: int, total_successful_attacks: int, total_images: int):
        self.source_model = source_model
        self.target_model = target_model
        self.attack_name = attack_name
        self.transfer_success = transfer_success
        self.total_successful_attacks = total_successful_attacks
        self.total_images = total_images
        self.transfer_rate = transfer_success / total_successful_attacks if total_successful_attacks > 0 else 0.0
        self.attack_success_rate = total_successful_attacks / total_images if total_images > 0 else 0.0


class TransferabilityLogger:
    """Class to handle logging and CSV saving for transferability experiments"""
    
    def __init__(self, results_folder: str = "results/transferability"):
        self.results_folder = results_folder
        self.failure_logs_folder = os.path.join(results_folder, "failure_logs")
        os.makedirs(self.results_folder, exist_ok=True)
        os.makedirs(self.failure_logs_folder, exist_ok=True)
        self._csv_path: Optional[str] = None
        self._recorded_keys: Set[Tuple[str, str, str]] = set()
    
    def begin_incremental_csv(self, filename: str, method: str) -> str:
        """
        Set the CSV path for this run. Creates the file with a header if it
        does not exist or is empty (fixed basename per experiment type).
        """
        self._csv_path = _transferability_csv_path(self.results_folder, filename, method)
        if (not os.path.isfile(self._csv_path)) or os.path.getsize(self._csv_path) == 0:
            with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=TRANSFERABILITY_CSV_FIELDNAMES)
                w.writeheader()
            self._recorded_keys = set()
        else:
            self._recorded_keys = _load_recorded_transfer_keys(self._csv_path)
        print(f"📄 Results CSV: {self._csv_path}")
        return self._csv_path
    
    def has_recorded(self, source_model: str, target_model: str, attack_name: str) -> bool:
        """True if the results CSV already has a row for this source, target, and attack."""
        return (source_model, target_model, attack_name) in self._recorded_keys
    
    def print_duplicate_row_skip(
        self, source_model: str, target_model: str, attack_name: str
    ) -> None:
        """Console only: explain why a (source, target, attack) row is not written again."""
        print(
            f"⏭️ Skip (already in results CSV): {source_model} → "
            f"{target_model} ({attack_name})"
        )
    
    def append_result(self, result: TransferabilityResult) -> None:
        """Open CSV, append one row (with row timestamp), close."""
        if not self._csv_path:
            raise RuntimeError("append_result called before begin_incremental_csv")
        key = (result.source_model, result.target_model, result.attack_name)
        if key in self._recorded_keys:
            self.print_duplicate_row_skip(
                result.source_model, result.target_model, result.attack_name
            )
            return
        row_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = {
            "source_model": result.source_model,
            "target_model": result.target_model,
            "attack_name": result.attack_name,
            "total_images": result.total_images,
            "total_successful_attacks": result.total_successful_attacks,
            "transfer_success": result.transfer_success,
            "transfer_rate": result.transfer_rate,
            "attack_success_rate": result.attack_success_rate,
            "timestamp": row_ts,
        }
        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=TRANSFERABILITY_CSV_FIELDNAMES).writerow(row)
        self._recorded_keys.add(key)
    
    def save_failure_log(self, source_model: str, target_model: str, attack_name: str, 
                        exception: Exception, method: str):
        """Save failure log to file"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"failure_log_{source_model}_{target_model}_{attack_name}_{method}_{timestamp}.txt"
        filepath = os.path.join(self.failure_logs_folder, filename)
        
        with open(filepath, 'w') as f:
            f.write("TRANSFERABILITY FAILURE LOG\n")
            f.write("=" * 50 + "\n")
            f.write(f"Source Model: {source_model}\n")
            f.write(f"Target Model: {target_model}\n")
            f.write(f"Attack: {attack_name}\n")
            f.write(f"Method: {method}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Exception Type: {type(exception).__name__}\n")
            f.write(f"Exception Message: {str(exception)}\n\n")
            f.write("Full Traceback:\n")
            f.write("=" * 50 + "\n")
            f.write(traceback.format_exc())
    
    def save_results_to_csv(self, results: List[TransferabilityResult], 
                           filename: str, method: str):
        """Write all results in one shot (each row includes its own timestamp)."""
        csv_path = _transferability_csv_path(self.results_folder, filename, method)
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=TRANSFERABILITY_CSV_FIELDNAMES)
            writer.writeheader()
            for result in results:
                row_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow(
                    {
                        "source_model": result.source_model,
                        "target_model": result.target_model,
                        "attack_name": result.attack_name,
                        "total_images": result.total_images,
                        "total_successful_attacks": result.total_successful_attacks,
                        "transfer_success": result.transfer_success,
                        "transfer_rate": result.transfer_rate,
                        "attack_success_rate": result.attack_success_rate,
                        "timestamp": row_ts,
                    }
                )
        print(f"✅ Results saved to: {csv_path}")


def imagenette_transferability_model2model_in_memory(
    model_names: List[str], 
    attack_names: List[str], 
    images_per_attack: int = 100,
    batch_size: int = 1,
    results_folder: str = "results/transferability"
) -> List[TransferabilityResult]:
    """
    In-memory transferability: model to model
    
    Args:
        model_names: List of model names to test
        attack_names: List of attack names to test
        images_per_attack: Number of images to attack per model-attack combination
        batch_size: Batch size for data loading
        results_folder: Folder to save results (incremental CSV per experiment type; each row open/append/close)
    
    Returns:
        List of TransferabilityResult objects
    """
    print("🔄 Starting in-memory model-to-model transferability analysis...")
    
    logger = TransferabilityLogger(results_folder)
    results = []
    logger.begin_incremental_csv("model2model_transferability", "in_memory")
    
    # Load test data
    _, test_loader = load_imagenette(batch_size=batch_size, test_subset_size=-1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for source_model_name in tqdm(model_names, desc="Source Models"):
        try:
            # Load source model
            source_model_path = f"./models/imagenette/{source_model_name}_advanced.pt"
            source_result = load_model_imagenette(source_model_path, source_model_name, device=device)
            source_model = source_result.model
            source_model.eval()

            for attack_name in tqdm(attack_names, desc=f"Attacks on {source_model_name}", leave=False):
                try:
                    pending_targets: List[str] = []
                    for t in model_names:
                        if t == source_model_name:
                            continue
                        if logger.has_recorded(source_model_name, t, attack_name):
                            logger.print_duplicate_row_skip(
                                source_model_name, t, attack_name
                            )
                        else:
                            pending_targets.append(t)
                    if not pending_targets:
                        continue
                    # Create attack on source model
                    attack = AttackFactory.get_attack(attack_name, source_model)
                    
                    # Generate adversarial examples on source model
                    adversarial_examples = []
                    source_labels = []
                    successful_attacks_count = 0
                    
                    for images, labels in test_loader:
                        if successful_attacks_count >= images_per_attack:
                            break
                            
                        images, labels = images.to(device), labels.to(device)
                        
                        # Remove misclassified images first
                        images, labels = ModelUtils.remove_missclassified_imagenette(
                            source_model, images, labels
                        )
                        
                        if labels.numel() == 0:
                            continue
                        
                        # Generate adversarial examples
                        adv_images = attack(images, labels)
                        
                        # Check source model success (attacks that fooled the source model)
                        with torch.no_grad():
                            source_outputs = source_model(adv_images)
                            source_predictions = torch.argmax(source_outputs, dim=1)
                            
                            # Process each image individually to count properly
                            for i in range(len(adv_images)):
                                if successful_attacks_count >= images_per_attack:
                                    break
                                    
                                label = labels[i].item()
                                predicted_label = source_predictions[i].item()
                                
                                # Only keep successful attacks (misclassified by source model)
                                if predicted_label != label:
                                    adversarial_examples.append(adv_images[i].cpu())
                                    source_labels.append(label)
                                    successful_attacks_count += 1
                    
                    if successful_attacks_count == 0:
                        print(f"⚠️ No successful attacks for {source_model_name} with {attack_name}")
                        continue
                    
                    # Stack all successful adversarial examples
                    if adversarial_examples:
                        all_adv_images = torch.stack(adversarial_examples)
                        all_source_labels = torch.tensor(source_labels)
                        
                        # Test transferability to other models
                        for target_model_name in pending_targets:
                            try:
                                # Load target model
                                target_model_path = f"./models/imagenette/{target_model_name}_advanced.pt"
                                target_result = load_model_imagenette(target_model_path, target_model_name, device=device)
                                target_model = target_result.model
                                
                                # Test transferability
                                transfer_success_count = 0
                                
                                with torch.no_grad():
                                    target_outputs = target_model(all_adv_images.to(device))
                                    target_predictions = torch.argmax(target_outputs, dim=1)
                                    
                                    # Count successful transfers (misclassified by target model)
                                    transfer_mask = target_predictions != all_source_labels.to(device)
                                    transfer_success_count = transfer_mask.sum().item()
                                
                                # Create result
                                result = TransferabilityResult(
                                    source_model=source_model_name,
                                    target_model=target_model_name,
                                    attack_name=attack_name,
                                    transfer_success=transfer_success_count,
                                    total_successful_attacks=len(all_adv_images),
                                    total_images=successful_attacks_count
                                )
                                
                                results.append(result)
                                logger.append_result(result)
                                
                                print(f"✅ {source_model_name} → {target_model_name} ({attack_name}): "
                                      f"{transfer_success_count}/{len(all_adv_images)} "
                                      f"({result.transfer_rate:.2%})")
                                
                            except Exception as e:
                                print(f"❌ Error testing {source_model_name} → {target_model_name} ({attack_name}): {e}")
                                logger.save_failure_log(source_model_name, target_model_name, attack_name, e, "model2model")
                                continue
                    
                except Exception as e:
                    print(f"❌ Error with attack {attack_name} on {source_model_name}: {e}")
                    logger.save_failure_log(source_model_name, "N/A", attack_name, e, "model2model")
                    continue
        
        except Exception as e:
            print(f"❌ Error loading source model {source_model_name}: {e}")
            logger.save_failure_log(source_model_name, "N/A", "N/A", e, "model2model")
            continue
    
    return results


def imagenette_transferability_attack2model_in_memory(
    model_names: List[str], 
    attack_names: List[str], 
    images_per_attack: int = 100,
    batch_size: int = 1,
    results_folder: str = "results/transferability"
) -> List[TransferabilityResult]:
    """
    In-memory transferability: attack to model (showing transferability for each attack to different models)
    
    Args:
        model_names: List of model names to test
        attack_names: List of attack names to test
        images_per_attack: Number of images to attack per model-attack combination
        batch_size: Batch size for data loading
        results_folder: Folder to save results (incremental CSV per experiment type; each row open/append/close)
    
    Returns:
        List of TransferabilityResult objects
    """
    print("🔄 Starting in-memory attack-to-model transferability analysis...")
    
    logger = TransferabilityLogger(results_folder)
    results = []
    logger.begin_incremental_csv("attack2model_transferability", "in_memory")
    _, test_loader = load_imagenette(batch_size=batch_size, test_subset_size=-1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for attack_name in tqdm(attack_names, desc="Attacks"):
        try:
            # Generate adversarial examples using the first model as source
            source_model_name = model_names[0]
            pending_targets = []
            for t in model_names:
                if logger.has_recorded(source_model_name, t, attack_name):
                    logger.print_duplicate_row_skip(source_model_name, t, attack_name)
                else:
                    pending_targets.append(t)
            if not pending_targets:
                continue
            source_model_path = f"./models/imagenette/{source_model_name}_advanced.pt"
            source_result = load_model_imagenette(source_model_path, source_model_name, device=device)
            source_model = source_result.model
            source_model.eval()

            # Create attack
            attack = AttackFactory.get_attack(attack_name, source_model)
            
            # Generate adversarial examples
            adversarial_examples = []
            source_labels = []
            successful_attacks_count = 0
            
            for images, labels in test_loader:
                if successful_attacks_count >= images_per_attack:
                    break
                    
                images, labels = images.to(device), labels.to(device)
                
                # Remove misclassified images first
                images, labels = ModelUtils.remove_missclassified_imagenette(
                    source_model, images, labels
                )
                
                if labels.numel() == 0:
                    continue
                
                # Generate adversarial examples
                adv_images = attack(images, labels)
                
                # Check source model success
                with torch.no_grad():
                    source_outputs = source_model(adv_images)
                    source_predictions = torch.argmax(source_outputs, dim=1)
                    
                    # Process each image individually to count properly
                    for i in range(len(adv_images)):
                        if successful_attacks_count >= images_per_attack:
                            break
                            
                        label = labels[i].item()
                        predicted_label = source_predictions[i].item()
                        
                        # Only keep successful attacks (misclassified by source model)
                        if predicted_label != label:
                            adversarial_examples.append(adv_images[i].cpu())
                            source_labels.append(label)
                            successful_attacks_count += 1
            
            if successful_attacks_count == 0:
                print(f"⚠️ No successful attacks for {attack_name}")
                continue
            
            # Stack all successful adversarial examples
            if adversarial_examples:
                all_adv_images = torch.stack(adversarial_examples)
                all_source_labels = torch.tensor(source_labels)
                
                # Test transferability to all models
                for target_model_name in tqdm(pending_targets, desc=f"Testing {attack_name}", leave=False):
                    try:
                        # Load target model
                        target_model_path = f"./models/imagenette/{target_model_name}_advanced.pt"
                        target_result = load_model_imagenette(target_model_path, target_model_name, device=device)
                        target_model = target_result.model
                        target_model.eval()
                        
                        # Test transferability
                        transfer_success_count = 0
                        
                        with torch.no_grad():
                            target_outputs = target_model(all_adv_images.to(device))
                            target_predictions = torch.argmax(target_outputs, dim=1)
                            
                            # Count successful transfers
                            transfer_mask = target_predictions != all_source_labels.to(device)
                            transfer_success_count = transfer_mask.sum().item()
                        
                        # Create result
                        result = TransferabilityResult(
                            source_model=source_model_name,
                            target_model=target_model_name,
                            attack_name=attack_name,
                            transfer_success=transfer_success_count,
                            total_successful_attacks=len(all_adv_images),
                            total_images=successful_attacks_count
                        )
                        
                        results.append(result)
                        logger.append_result(result)
                        
                        print(f"✅ {attack_name} → {target_model_name}: "
                              f"{transfer_success_count}/{len(all_adv_images)} "
                              f"({result.transfer_rate:.2%})")
                        
                    except Exception as e:
                        print(f"❌ Error testing {attack_name} → {target_model_name}: {e}")
                        logger.save_failure_log(source_model_name, target_model_name, attack_name, e, "attack2model")
                        continue
        
        except Exception as e:
            print(f"❌ Error with attack {attack_name}: {e}")
            logger.save_failure_log("N/A", "N/A", attack_name, e, "attack2model")
            continue
    
    return results


def imagenette_transferability_model2model_from_files(
    model_names: List[str], 
    attack_names: List[str], 
    attacked_images_folder: str = "data/attacks/imagenette_models",
    results_folder: str = "results/transferability"
) -> List[TransferabilityResult]:
    """
    File-based transferability: model to model (using saved adversarial images)
    
    Args:
        model_names: List of model names to test
        attack_names: List of attack names to test
        attacked_images_folder: Root folder; images may live under train/ or test/
            subfolders (as from imagenette_adv_imgs_generator) or under model/attack/ (legacy).
        results_folder: Folder to save results (incremental CSV per experiment type; each row open/append/close)
    
    Returns:
        List of TransferabilityResult objects
    """
    print("🔄 Starting file-based model-to-model transferability analysis...")
    
    logger = TransferabilityLogger(results_folder)
    results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.begin_incremental_csv("model2model_transferability", "from_files")
    
    for source_model_name in tqdm(model_names, desc="Source Models"):
        for attack_name in tqdm(attack_names, desc=f"Attacks on {source_model_name}", leave=False):
            try:
                pending_targets = []
                for t in model_names:
                    if t == source_model_name:
                        continue
                    if logger.has_recorded(source_model_name, t, attack_name):
                        logger.print_duplicate_row_skip(
                            source_model_name, t, attack_name
                        )
                    else:
                        pending_targets.append(t)
                if not pending_targets:
                    continue
                # Path to saved adversarial images
                adv_images_path = os.path.join(attacked_images_folder, source_model_name, attack_name)
                
                if not os.path.exists(adv_images_path):
                    print(f"⚠️ No saved images found for {source_model_name}/{attack_name}")
                    continue
                
                # Load adversarial images from files
                adversarial_examples = []
                source_labels = []
                
                # Get all class folders
                class_folders = [d for d in os.listdir(adv_images_path) 
                               if os.path.isdir(os.path.join(adv_images_path, d))]
                
                for class_folder in class_folders:
                    class_path = os.path.join(adv_images_path, class_folder)
                    class_label = int(class_folder)  # Assuming folder names are class labels
                    
                    # Load adversarial images (skip clean images)
                    image_files = [f for f in os.listdir(class_path) 
                                 if f.endswith('.png') and not f.startswith('src_')]
                    
                    for image_file in image_files:
                        image_path = os.path.join(class_path, image_file)
                        try:
                            # Load image and convert to tensor
                            image = Image.open(image_path).convert('RGB')
                            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
                            adversarial_examples.append(image_tensor)
                            source_labels.append(class_label)
                        except Exception as e:
                            print(f"⚠️ Error loading {image_path}: {e}")
                            continue
                
                if not adversarial_examples:
                    print(f"⚠️ No valid adversarial images found for {source_model_name}/{attack_name}")
                    continue
                
                # Convert to tensors
                all_adv_images = torch.stack(adversarial_examples)
                all_source_labels = torch.tensor(source_labels)
                successful_attacks_count = len(adversarial_examples)
                
                print(f"📁 Loaded {successful_attacks_count} adversarial images for {source_model_name}/{attack_name}")
                
                # Test transferability to other models
                for target_model_name in pending_targets:
                    try:
                        # Load target model
                        target_model_path = f"./models/imagenette/{target_model_name}_advanced.pt"
                        target_result = load_model_imagenette(target_model_path, target_model_name, device=device)
                        if not target_result.success:
                            print(
                                f"❌ Failed to load target {target_model_name}: {(target_result.error or 'unknown')}"
                            )
                            continue
                        target_model = target_result.model
                        target_model.eval()
                        
                        # Test transferability
                        transfer_success_count = 0
                        
                        # Process in batches to avoid memory issues
                        batch_size = 32
                        for i in range(0, len(all_adv_images), batch_size):
                            batch_adv = all_adv_images[i:i+batch_size].to(device)
                            batch_labels = all_source_labels[i:i+batch_size].to(device)
                            
                            with torch.no_grad():
                                target_outputs = target_model(batch_adv)
                                target_predictions = torch.argmax(target_outputs, dim=1)
                                
                                # Count successful transfers
                                transfer_mask = target_predictions != batch_labels
                                transfer_success_count += transfer_mask.sum().item()
                        
                        # Create result
                        result = TransferabilityResult(
                            source_model=source_model_name,
                            target_model=target_model_name,
                            attack_name=attack_name,
                            transfer_success=transfer_success_count,
                            total_successful_attacks=len(all_adv_images),
                            total_images=len(all_adv_images)  # For file-based, total = successful since files only contain successful attacks
                        )
                        
                        results.append(result)
                        logger.append_result(result)
                        
                        print(f"✅ {source_model_name} → {target_model_name} ({attack_name}): "
                              f"{transfer_success_count}/{len(all_adv_images)} "
                              f"({result.transfer_rate:.2%})")
                        
                    except Exception as e:
                        print(f"❌ Error testing {source_model_name} → {target_model_name} ({attack_name}): {e}")
                        logger.save_failure_log(source_model_name, target_model_name, attack_name, e, "model2model_files")
                        continue
            
            except Exception as e:
                print(f"❌ Error processing {source_model_name}/{attack_name}: {e}")
                logger.save_failure_log(source_model_name, "N/A", attack_name, e, "model2model_files")
                continue
    
    return results


def imagenette_transferability_attack2model_from_files(
    model_names: List[str], 
    attack_names: List[str], 
    attacked_images_folder: str = "data/attacks/imagenette_models",
    results_folder: str = "results/transferability"
) -> List[TransferabilityResult]:
    """
    File-based transferability: attack to model (using saved adversarial images)
    
    Args:
        model_names: List of model names to test
        attack_names: List of attack names to test
        attacked_images_folder: Root folder; images may live under train/ or test/
            subfolders (as from imagenette_adv_imgs_generator) or under model/attack/ (legacy).
        results_folder: Folder to save results (incremental CSV per experiment type; each row open/append/close)
    
    Returns:
        List of TransferabilityResult objects
    """
    print("🔄 Starting file-based attack-to-model transferability analysis...")
    
    logger = TransferabilityLogger(results_folder)
    results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.begin_incremental_csv("attack2model_transferability", "from_files")
    
    for attack_name in tqdm(attack_names, desc="Attacks"):
        try:
            # Use first model as source for adversarial examples
            source_model_name = model_names[0]
            pending_targets = []
            for t in model_names:
                if logger.has_recorded(source_model_name, t, attack_name):
                    logger.print_duplicate_row_skip(source_model_name, t, attack_name)
                else:
                    pending_targets.append(t)
            if not pending_targets:
                continue
            adv_images_path = _resolve_saved_adv_images_dir(
                attacked_images_folder, source_model_name, attack_name
            )
            if adv_images_path is None:
                print(
                    f"⚠️ No saved images found for {source_model_name}/{attack_name} "
                    f"(tried train/, test/, and flat layout under {attacked_images_folder})"
                )
                continue
            
            # Load adversarial images from files
            adversarial_examples = []
            source_labels = []
            
            # Get all class folders
            class_folders = [d for d in os.listdir(adv_images_path) 
                           if os.path.isdir(os.path.join(adv_images_path, d))]
            
            for class_folder in class_folders:
                class_path = os.path.join(adv_images_path, class_folder)
                class_label = int(class_folder)
                
                # Load adversarial images (skip clean images)
                image_files = [f for f in os.listdir(class_path) 
                             if f.endswith('.png') and not f.startswith('src_')]
                
                for image_file in image_files:
                    image_path = os.path.join(class_path, image_file)
                    try:
                        # Load image and convert to tensor
                        image = Image.open(image_path).convert('RGB')
                        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
                        adversarial_examples.append(image_tensor)
                        source_labels.append(class_label)
                    except Exception as e:
                        print(f"⚠️ Error loading {image_path}: {e}")
                        continue
            
            if not adversarial_examples:
                print(f"⚠️ No valid adversarial images found for {attack_name}")
                continue
            
            # Convert to tensors
            all_adv_images = torch.stack(adversarial_examples)
            all_source_labels = torch.tensor(source_labels)
            successful_attacks_count = len(adversarial_examples)
            
            print(f"📁 Loaded {successful_attacks_count} adversarial images for {attack_name}")
            
            # Test transferability to all models
            for target_model_name in tqdm(pending_targets, desc=f"Testing {attack_name}", leave=False):
                try:
                    # Load target model
                    target_model_path = f"./models/imagenette/{target_model_name}_advanced.pt"
                    target_result = load_model_imagenette(target_model_path, target_model_name, device=device)
                    if not target_result.success:
                        print(
                            f"❌ Failed to load target {target_model_name}: {(target_result.error or 'unknown')}"
                        )
                        continue
                    target_model = target_result.model
                    target_model.eval()
                    
                    # Test transferability
                    transfer_success_count = 0
                    
                    # Process in batches
                    batch_size = 32
                    for i in range(0, len(all_adv_images), batch_size):
                        batch_adv = all_adv_images[i:i+batch_size].to(device)
                        batch_labels = all_source_labels[i:i+batch_size].to(device)
                        
                        with torch.no_grad():
                            target_outputs = target_model(batch_adv)
                            target_predictions = torch.argmax(target_outputs, dim=1)
                            
                            # Count successful transfers
                            transfer_mask = target_predictions != batch_labels
                            transfer_success_count += transfer_mask.sum().item()
                    
                    result = TransferabilityResult(
                        source_model=source_model_name,
                        target_model=target_model_name,
                        attack_name=attack_name,
                        transfer_success=transfer_success_count,
                        total_successful_attacks=len(all_adv_images),
                        total_images=successful_attacks_count  # For file-based, total = successful since files only contain successful attacks
                    )
                    
                    results.append(result)
                    logger.append_result(result)
                    
                    print(f"✅ {attack_name} → {target_model_name}: "
                          f"{transfer_success_count}/{len(all_adv_images)} "
                          f"({result.transfer_rate:.2%})")
                    
                except Exception as e:
                    print(f"❌ Error testing {attack_name} → {target_model_name}: {e}")
                    logger.save_failure_log(source_model_name, target_model_name, attack_name, e, "attack2model_files")
                    continue
        
        except Exception as e:
            print(f"❌ Error with attack {attack_name}: {e}")
            logger.save_failure_log("N/A", "N/A", attack_name, e, "attack2model_files")
            continue
    
    return results


def run_all_transferability_experiments(
    model_names: Optional[List[str]] = None,
    attack_names: Optional[List[str]] = None,
    images_per_attack: int = 100,
    attacked_images_folder: str = "data/attacks/imagenette_models",
    results_folder: str = "results/transferability"
):
    """
    Run all transferability experiments
    
    Args:
        model_names: List of model names (default: all available models)
        attack_names: List of attack names (default: all available attacks)
        images_per_attack: Number of images per attack for in-memory experiments
        attacked_images_folder: Folder containing saved adversarial images
        results_folder: Folder to save results
    """
    print("🚀 Starting comprehensive transferability analysis...")
    
    # Set default model and attack names if not provided
    if model_names is None:
        model_names = [ModelNames().resnet18, ModelNames().densenet121, 
                      ModelNames().mobilenet_v2, ModelNames().efficientnet_b0]
    
    if attack_names is None:
        attack_names = [AttackNames().FGSM, AttackNames().PGD, AttackNames().BIM, 
                       AttackNames().FFGSM, AttackNames().TPGD]
    
    print(f"📊 Models: {model_names}")
    print(f"🎯 Attacks: {attack_names}")
    
    # Run all experiments
    experiments = [
        ("In-Memory Model-to-Model", lambda: imagenette_transferability_model2model_in_memory(
            model_names, attack_names, images_per_attack, results_folder=results_folder)),
        ("In-Memory Attack-to-Model", lambda: imagenette_transferability_attack2model_in_memory(
            model_names, attack_names, images_per_attack, results_folder=results_folder)),
        ("File-Based Model-to-Model", lambda: imagenette_transferability_model2model_from_files(
            model_names, attack_names, attacked_images_folder, results_folder=results_folder)),
        ("File-Based Attack-to-Model", lambda: imagenette_transferability_attack2model_from_files(
            model_names, attack_names, attacked_images_folder, results_folder=results_folder))
    ]
    
    all_results = {}
    
    for exp_name, exp_func in experiments:
        print(f"\n{'='*60}")
        print(f"🔄 Running: {exp_name}")
        print(f"{'='*60}")
        
        try:
            results = exp_func()
            all_results[exp_name] = results
            print(f"✅ {exp_name} completed successfully!")
            
        except Exception as e:
            print(f"❌ {exp_name} failed: {e}")
            continue
    
    print(f"\n🎉 All transferability experiments completed!")
    print(f"📁 Results saved in: {results_folder}")
    
    return all_results


if __name__ == "__main__":
    # Example usage
    model_names = [
        ModelNames().resnet18,
        ModelNames().densenet121,
        ModelNames().mobilenet_v2,
        ModelNames().efficientnet_b0,
        ModelNames().vgg16,
    ]
    
    attack_names = AttackNames().all_attack_names
    
    # Run all experiments
    save_path = "results/imagenette/transferability/model2model"
    imagenette_transferability_model2model_in_memory(model_names, attack_names, images_per_attack=500, results_folder=save_path)

    save_path = "results/imagenette/transferability/attack2model"
    imagenette_transferability_attack2model_in_memory(model_names, attack_names, images_per_attack=500, results_folder=save_path)