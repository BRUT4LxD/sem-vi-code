import datetime
import os
import time
import traceback
import torch
import csv
from typing import Callable, Dict, List, Optional, Set
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


DIRECT_ATTACK_CSV_FIELDNAMES: List[str] = [
    'model_name',
    'attack_name',
    'acc',
    'prec',
    'rec',
    'f1',
    'L0_pixels',
    'L1',
    'L2',
    'Linf',
    'power_mse',
    'clean_accuracy',
    'AD',
    'RAD',
    'asr_unconditional',
    'time',
    'n_samples',
]


class DirectAttackLogger:
    """Handles incremental CSV writes for direct attack experiments."""

    def __init__(self, results_folder: str):
        self.results_folder = results_folder
        self._csv_paths: Dict[str, str] = {}
        self._recorded_attacks: Dict[str, Set[str]] = {}
        os.makedirs(results_folder, exist_ok=True)

    def begin_incremental_csv(self, model_name: str) -> str:
        csv_path = os.path.join(self.results_folder, f"{model_name}.csv")
        self._csv_paths[model_name] = csv_path

        if (not os.path.isfile(csv_path)) or os.path.getsize(csv_path) == 0:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=DIRECT_ATTACK_CSV_FIELDNAMES)
                writer.writeheader()
            self._recorded_attacks[model_name] = set()
        else:
            self._recorded_attacks[model_name] = self._load_recorded_attack_names(csv_path)

        print(f"📄 Results CSV for {model_name}: {csv_path}")
        return csv_path

    def has_recorded(self, model_name: str, attack_name: str) -> bool:
        return attack_name in self._recorded_attacks.get(model_name, set())

    def append_result(self, result: dict) -> None:
        model_name = result['model_name']
        attack_name = result['attack_name']
        csv_path = self._csv_paths.get(model_name)
        if csv_path is None:
            raise RuntimeError(f"append_result called before begin_incremental_csv for {model_name}")

        if self.has_recorded(model_name, attack_name):
            print(f"    ⏭️  Skipping duplicate row for {model_name}/{attack_name}")
            return

        filtered_result = {key: value for key, value in result.items() if key != 'conf_matrix'}
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=DIRECT_ATTACK_CSV_FIELDNAMES)
            writer.writerow(filtered_result)
        self._recorded_attacks[model_name].add(attack_name)
        print(f"    💾 Appended result to: {csv_path}")

    @staticmethod
    def _load_recorded_attack_names(csv_path: str) -> Set[str]:
        recorded: Set[str] = set()
        try:
            with open(csv_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    return recorded
                for row in reader:
                    attack_name = row.get('attack_name', '').strip()
                    if attack_name:
                        recorded.add(attack_name)
        except OSError:
            pass
        return recorded


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
        checkpoint_path_for_model: Optional[Callable[[str], str]] = None,
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
            models_to_run: List[LoadedModel] = list(loaded_models)
        elif model_names is not None:
            models_to_run = []
            for name in model_names:
                if checkpoint_path_for_model is not None:
                    model_path = checkpoint_path_for_model(name)
                else:
                    model_path = f"./models/imagenette/{name}_advanced.pt"
                lm = load_model_imagenette(model_path, name, device=self.device_name)
                if not lm.success:
                    print(f"❌ Failed to load model {name}: {lm.error}")
                    continue
                models_to_run.append(lm)
        else:
            raise ValueError("Either model_names or loaded_models must be provided")

        save_results = results_folder != ""

        logger = DirectAttackLogger(results_folder) if save_results else None

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
            if logger is not None:
                logger.begin_incremental_csv(model_name)

            # Store results for this model
            model_results = []

            # Run each attack on this model
            for attack_name in attack_names:
                if logger is not None and logger.has_recorded(model_name, attack_name):
                    print(f"  ⏭️  Skipping {attack_name} (already in CSV)")
                    continue
                print(f"  ⚔️  Running {attack_name} attack...")

                try:
                    attack = AttackFactory.get_attack(attack_name, model)
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
                        result_dict['L0_pixels'] = distance_score.l0_pixels
                        result_dict['L1'] = distance_score.l1
                        result_dict['L2'] = distance_score.l2
                        result_dict['Linf'] = distance_score.linf
                        result_dict['power_mse'] = distance_score.power_mse

                        # Add accuracy drop metrics
                        result_dict['clean_accuracy'] = ev.clean_accuracy
                        result_dict['AD'] = ev.accuracy_drop
                        result_dict['RAD'] = ev.relative_accuracy_drop
                        result_dict['asr_unconditional'] = ev.asr_unconditional

                        result_dict['time'] = ev.time
                        result_dict['n_samples'] = ev.n_samples

                        model_results.append(result_dict)
                        if logger is not None:
                            logger.append_result(result_dict)
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
    checkpoint_path_for_model: Optional[Callable[[str], str]] = None,
):
    direct_attacks = ImageNetteDirectAttacks(device=device)
    return direct_attacks.run_attacks_on_models(
        attack_names=attack_names,
        data_loader=data_loader,
        model_names=model_names,
        loaded_models=loaded_models,
        results_folder=results_folder,
        checkpoint_path_for_model=checkpoint_path_for_model,
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


if __name__ == "__main__":
    import glob
    from attacks.attack_names import AttackNames
    from data_eng.dataset_loader import load_imagenette
    from domain.model.model_names import ModelNames

    progressive_dir = "./models/imagenette_adversarial_progressive"
    model_files = sorted(glob.glob(os.path.join(progressive_dir, "*.pt")))

    model_files = [
        # './models/imagenette_adversarial/densenet121_adv_preattacked_20260415.pt',
        # './models/imagenette_adversarial/efficientnet_b0_adv_preattacked_20260415.pt',
        # './models/imagenette_adversarial/mobilenet_v2_adv_preattacked_20260415.pt',
        # './models/imagenette_adversarial/resnet18_adv_preattacked_20260415.pt',
        './models/imagenette_adversarial/resnet18_adv_preattacked_20260415.pt',
        ]
    print(f"Model files: {model_files}")

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

