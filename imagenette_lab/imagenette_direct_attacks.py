import datetime
import os
import time
import traceback
import torch
import csv
from typing import List, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
from attacks.attack import Attack
from attacks.system_under_attack import SystemUnderAttack
from attacks.attack_factory import AttackFactory
from config.imagenette_classes import ImageNetteClasses
from data_eng.io import load_model_imagenette
from domain.attack.attack_result import AttackResult
from domain.model.loaded_model import LoadedModel
from evaluation.metrics import Metrics


def normalize_adversarial_image(adv_image: torch.Tensor) -> torch.Tensor:
    """
    Normalize adversarial image to [0,1] range if it exceeds boundaries.
    Some attacks may produce values outside [0,1] range, this function clamps them.
    
    Args:
        adv_image: Adversarial image tensor
        
    Returns:
        Normalized image tensor in [0,1] range
    """
    return torch.clamp(adv_image, min=0.0, max=1.0)


class ImageNetteDirectAttacks:
    """
    Helper class for running direct attacks on ImageNette models.
    """

    def __init__(self, device: str = 'auto'):
        if device == 'auto':
            self.device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device_name = device
        self.device = torch.device(self.device_name)

    def attack_images_imagenette(self, attack: Attack, data_loader: DataLoader):
        model = attack.model

        model.eval()
        model.to(self.device)

        num_classes = len(ImageNetteClasses.get_classes())

        attack_results = []
        start_time = time.time()

        clean_correct = 0
        clean_total = 0

        for images, labels in tqdm(data_loader):
            # 1. Remove missclassified images
            # 2. Attack the images
            # 3. Test the attacked images
            # 4. Collect the evaluation scores
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # Calculate clean accuracy
            clean_total += labels.size(0)
            clean_correct += (predictions == labels).sum().item()

            # Remove missclassified images
            images, labels = images[predictions == labels], labels[predictions == labels]

            if labels.numel() == 0:
                continue

            adv_images = attack(images, labels)

            # Normalize adversarial images to [0,1] range (some attacks may exceed boundaries)
            adv_images = normalize_adversarial_image(adv_images)

            with torch.no_grad():
                outputs = model(adv_images)
                _, predicted_labels = torch.max(outputs.data, 1)
                for i in range(len(adv_images)):
                    label = labels[i].item()
                    predicted_label = predicted_labels[i].item()

                    att_result = AttackResult(
                        actual=label,
                        predicted=predicted_label,
                        adv_image=adv_images[i].cpu(),
                        src_image=images[i].cpu(),
                        model_name=attack.model_name,
                        attack_name=attack.attack)

                    attack_results.append(att_result)

        # 5. Calculate global evaluation scores
        if len(attack_results) == 0:
            print("Warning: No successful attacks generated")
            return None

        clean_accuracy = 100.0 * clean_correct / clean_total if clean_total > 0 else 0.0
        ev = Metrics.evaluate_attack_score(attack_results, num_classes, clean_accuracy=clean_accuracy)
        ev.set_after_attack(time.time() - start_time, len(attack_results))

        return ev

    def run_attacks_on_models(
        self,
        attack_names: List[str],
        data_loader: DataLoader,
        model_names: Optional[List[str]] = None,
        loaded_models: Optional[List[LoadedModel]] = None,
        results_folder: str = "",
    ):
        """
        Run attacks on multiple models and save results to CSV files.

        Provide either ``loaded_models`` (pre-loaded :class:`LoadedModel` instances)
        or ``model_names`` (names resolved to ``./models/imagenette/{name}_advanced.pt``).
        If both are given, ``loaded_models`` takes precedence.

        Args:
            attack_names: List of attack algorithm names to run.
            data_loader: DataLoader with test images.
            model_names: Model name strings (used for auto-loading from default path).
            loaded_models: Pre-loaded LoadedModel instances (skips loading step).
            results_folder: Folder to save CSV results (empty string to skip saving).
        """
        if loaded_models is not None:
            models_to_run: List[LoadedModel] = loaded_models
        elif model_names is not None:
            models_to_run = []
            for name in model_names:
                model_path = f"./models/imagenette/{name}_advanced.pt"
                lm = load_model_imagenette(model_path, name, device=self.device_name)
                if not lm.success:
                    print(f"❌ Failed to load model {name}: {lm.error}")
                    continue
                models_to_run.append(lm)
        else:
            raise ValueError("Either model_names or loaded_models must be provided")

        save_results = results_folder != ""

        if save_results:
            os.makedirs(results_folder, exist_ok=True)

        print("🚀 Starting comprehensive attack evaluation...")
        print(f"📊 Models: {len(models_to_run)}")
        print(f"⚔️  Attacks: {len(attack_names)}")
        print(f"📁 Results will be saved to: {results_folder}")
        print("=" * 80)

        all_results = {}

        for loaded in models_to_run:
            model_name = loaded.model_name
            model = loaded.model
            print(f"\n🔍 Model: {model_name}")

            # Store results for this model
            model_results = []

            # Run each attack on this model
            for attack_name in attack_names:
                print(f"  ⚔️  Running {attack_name} attack...")

                try:
                    # Create attack
                    attack = AttackFactory.get_attack(attack_name, model)

                    # Run attack
                    ev = self.attack_images_imagenette(attack, data_loader)

                    if ev is not None:
                        result_dict = {}
                        result_dict['model_name'] = model_name
                        result_dict['attack_name'] = attack_name
                        result_dict['acc'] = ev.acc
                        result_dict['prec'] = ev.prec
                        result_dict['rec'] = ev.rec
                        result_dict['f1'] = ev.f1

                        distance_score = ev.distance_score
                        result_dict['l0_pixels'] = distance_score.l0_pixels
                        result_dict['l1'] = distance_score.l1
                        result_dict['l2'] = distance_score.l2
                        result_dict['linf'] = distance_score.linf
                        result_dict['power_mse'] = distance_score.power_mse

                        # Add accuracy drop metrics
                        result_dict['clean_accuracy'] = ev.clean_accuracy
                        result_dict['accuracy_drop'] = ev.accuracy_drop
                        result_dict['relative_accuracy_drop'] = ev.relative_accuracy_drop
                        result_dict['asr_unconditional'] = ev.asr_unconditional
                        result_dict['asr_conditional'] = ev.asr_conditional

                        result_dict['time'] = ev.time
                        result_dict['n_samples'] = ev.n_samples

                        model_results.append(result_dict)
                        print(f"    ✅ {attack_name}: {ev.acc:.2f}% accuracy")
                    else:
                        print(f"    ⚠️  {attack_name}: No results generated")

                except Exception as e:
                    print(f"    ❌ {attack_name} failed: {str(e)}")
                    if save_results:
                        save_failure_log(model_name, attack_name, e, results_folder)
                    continue

            # Store results for this model
            all_results[model_name] = model_results

            # Save individual model results to CSV
            if save_results and model_results:
                save_model_results_to_csv(model_name, model_results, results_folder)

        # Print summary
        print("\n" + "=" * 80)
        print("📋 SUMMARY")
        print("=" * 80)

        for model_name, results in all_results.items():
            print(f"\n🔍 {model_name}:")
            if results:
                for result in results:
                    attack_name = result['attack_name']
                    accuracy = result['acc']
                    print(f"  {attack_name:12s}: {accuracy:6.2f}% accuracy")
            else:
                print("  No successful results")

        return all_results


def attack_images_imagenette(attack: Attack, data_loader: DataLoader, successfully_attacked_images_folder: str = ""):
    direct_attacks = ImageNetteDirectAttacks(device='auto')
    return direct_attacks.attack_images_imagenette(attack, data_loader)


def run_attacks_on_models(
    attack_names: List[str],
    data_loader: DataLoader,
    model_names: Optional[List[str]] = None,
    loaded_models: Optional[List[LoadedModel]] = None,
    device: str = 'cuda',
    results_folder: str = "",
):
    direct_attacks = ImageNetteDirectAttacks(device=device)
    return direct_attacks.run_attacks_on_models(
        attack_names=attack_names,
        data_loader=data_loader,
        model_names=model_names,
        loaded_models=loaded_models,
        results_folder=results_folder,
    )


def save_failure_log(model_name: str, attack_name: str, exception: Exception, results_folder: str):
    """
    Save failure log to a file with detailed exception information.

    Args:
        model_name: Name of the model that failed
        attack_name: Name of the attack that failed
        exception: The exception that occurred
        results_folder: Folder to save the failure log
    """
    # Create failure logs directory
    failure_logs_dir = os.path.join(results_folder, "failure_logs")
    os.makedirs(failure_logs_dir, exist_ok=True)

    # Create filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"failure_log_{model_name}_{attack_name}_{timestamp}.txt"
    filepath = os.path.join(failure_logs_dir, filename)

    # Write failure log
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("FAILURE LOG\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Attack: {attack_name}\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Exception Type: {type(exception).__name__}\n")
        f.write(f"Exception Message: {str(exception)}\n")
        f.write("\nFull Traceback:\n")
        f.write("=" * 50 + "\n")
        f.write(traceback.format_exc())
        f.write("\n" + "=" * 50 + "\n")

    print(f"    📝 Failure log saved to: {filepath}")


def save_model_results_to_csv(model_name: str, results: List[dict], results_folder: str):
    """
    Save results for a single model to CSV file.
    
    Args:
        model_name: Name of the model
        results: List of result dictionaries
        results_folder: Folder to save the CSV file
    """
    
    if not results:
        return
    
    # Create CSV filename
    csv_filename = f"{model_name}.csv"
    csv_path = os.path.join(results_folder, csv_filename)
    
    # Get fieldnames from first result (excluding model_name and conf_matrix)
    fieldnames = [key for key in results[0].keys() if key not in ['model_name', 'conf_matrix']]
    # Put model_name first
    fieldnames = ['model_name'] + fieldnames
    
    # Write CSV file
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Write data rows (excluding conf_matrix)
        for result in results:
            filtered_result = {key: value for key, value in result.items() if key != 'conf_matrix'}
            writer.writerow(filtered_result)
    
    print(f"    💾 Saved results to: {csv_path}")


if __name__ == "__main__":
    import glob
    from attacks.attack_names import AttackNames
    from data_eng.dataset_loader import load_imagenette
    from domain.model.model_names import ModelNames

    progressive_dir = "./models/imagenette_adversarial_progressive"
    model_files = sorted(glob.glob(os.path.join(progressive_dir, "*.pt")))

    if not model_files:
        print(f"❌ No .pt files found in {progressive_dir}")
        exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    loaded_models: List[LoadedModel] = []
    for path in model_files:
        lm = load_model_imagenette(path, device=device)
        if lm.success:
            loaded_models.append(lm)

    attack_names = AttackNames().all_attack_names

    _, test_loader = load_imagenette(batch_size=4, test_subset_size=500)

    runner = ImageNetteDirectAttacks(device=device)
    runner.run_attacks_on_models(
        attack_names=attack_names,
        data_loader=test_loader,
        loaded_models=loaded_models,
        results_folder="./results/imagenette/progressive_attacks",
    )

