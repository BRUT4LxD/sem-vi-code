import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from attacks.attack_factory import AttackFactory
from attacks.attack_names import AttackNames
from config.imagenet_models import ImageNetModels
from data_eng.dataset_loader import load_imagenette
from domain.attack.attack_distance_score import AttackDistanceScore
from domain.attack.attack_result import AttackResult
from domain.model.model_names import ModelNames
from evaluation.metrics import Metrics
from imagenette_lab.imagenette_direct_attacks import normalize_adversarial_image
from imagenette_lab.training.imagenette_base_trainer import BaseImageNetteTrainer
from training.train import Training
from training.transfer.setup_pretraining import SetupPretraining


class ImageNetteAdversarialProgressiveTrainer(BaseImageNetteTrainer):
    """
    Progressive adversarial training for ImageNette.

    At each iteration, the trainer generates a fresh set of successful
    adversarial examples for every attack and appends them to the existing
    attacked dataset, so the training set keeps growing over time.
    """

    def __init__(self, device: str = "auto", models_dir: str = "./models/imagenette"):
        super().__init__(device=device, models_dir=models_dir)
        self.progressive_adversarial_models_dir = "./models/imagenette_adversarial_progressive"
        os.makedirs(self.progressive_adversarial_models_dir, exist_ok=True)

    @staticmethod
    def _save_attacked_image(
        adv_image: torch.Tensor,
        label: int,
        model_name: str,
        attack_name: str,
        attacked_images_folder: str,
        attack_test_dataset: bool,
        iteration: int,
    ) -> None:
        dataset_name = "test" if attack_test_dataset else "train"
        save_dir = os.path.join(
            attacked_images_folder,
            dataset_name,
            model_name,
            attack_name,
            str(label),
        )
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_name = f"progressive_iter{iteration + 1}_{timestamp}.png"
        ToPILImage()(adv_image.cpu()).save(os.path.join(save_dir, file_name))

    def _collect_successful_adversarial_examples(
        self,
        model: torch.nn.Module,
        attack_names: List[str],
        data_loader: DataLoader,
        images_per_attack: int,
        attack_test_dataset: bool,
        save_generated_images: bool,
        attacked_images_folder: str,
        iteration: int,
        verbose: bool,
    ) -> Tuple[List[Tuple[torch.Tensor, int]], AttackDistanceScore]:
        if images_per_attack <= 0:
            return [], AttackDistanceScore(0.0, 0.0, 0.0, 0.0, 0.0)

        model.eval()
        examples: List[Tuple[torch.Tensor, int]] = []
        attack_results: List[AttackResult] = []

        for attack_name in attack_names:
            attack = AttackFactory.get_attack(attack_name, model)
            successful_examples = 0
            attempted_examples = 0
            max_attempts = images_per_attack * 100

            progress_bar = tqdm(
                data_loader,
                desc=f"Generating {attack_name} ({'test' if attack_test_dataset else 'train'})",
                disable=not verbose,
                leave=False,
            )

            for images, labels in progress_bar:
                if successful_examples >= images_per_attack or attempted_examples >= max_attempts:
                    break

                images = images.to(self.device)
                labels = labels.to(self.device)

                with torch.no_grad():
                    predictions = torch.argmax(model(images), dim=1)

                images = images[predictions == labels]
                labels = labels[predictions == labels]

                if labels.numel() == 0:
                    continue

                adv_images = normalize_adversarial_image(attack(images, labels))

                with torch.no_grad():
                    adv_predictions = torch.argmax(model(adv_images), dim=1)

                for i in range(len(adv_images)):
                    if successful_examples >= images_per_attack or attempted_examples >= max_attempts:
                        break

                    attempted_examples += 1
                    label = labels[i].item()
                    if adv_predictions[i].item() == label:
                        continue

                    src_image = images[i].detach().cpu()
                    adv_image = adv_images[i].detach().cpu()
                    examples.append((adv_image, label))
                    attack_results.append(
                        AttackResult(
                            actual=label,
                            predicted=adv_predictions[i].item(),
                            adv_image=adv_image,
                            src_image=src_image,
                            model_name=model.__class__.__name__,
                            attack_name=attack_name,
                        )
                    )
                    successful_examples += 1

                    if save_generated_images:
                        self._save_attacked_image(
                            adv_image=adv_image,
                            label=label,
                            model_name=model.__class__.__name__,
                            attack_name=attack_name,
                            attacked_images_folder=attacked_images_folder,
                            attack_test_dataset=attack_test_dataset,
                            iteration=iteration,
                        )

                    progress_bar.set_postfix({"saved": successful_examples})

            if successful_examples < images_per_attack:
                cap_note = (
                    f" Reached attempt cap ({attempted_examples}/{max_attempts})."
                    if attempted_examples >= max_attempts
                    else ""
                )
                print(
                    f"⚠️ {attack_name}: generated only {successful_examples}/{images_per_attack} "
                    f"successful adversarial examples on the "
                    f"{'test' if attack_test_dataset else 'train'} split."
                    f"{cap_note}"
                )
            else:
                print(
                    f"✅ {attack_name}: generated {successful_examples}/{images_per_attack} "
                    f"successful adversarial examples."
                )

        return examples, Metrics.attack_distance_score(attack_results)

    def train_progressive_adversarial_model(
        self,
        model: torch.nn.Module = None,
        attack_names: List[str] = None,
        learning_rate: float = 0.001,
        iterations: int = 5,
        epochs_per_iteration: int = 2,
        batch_size: int = 32,
        images_per_attack_per_iteration: int = 10,
        validation_images_per_attack_per_iteration: Optional[int] = None,
        early_stopping_patience: int = 7,
        scheduler_type: str = "step",
        scheduler_params: dict = None,
        weight_decay: float = 0.0001,
        gradient_clip_norm: float = 1.0,
        save_model_path: str = None,
        writer: SummaryWriter = None,
        save_generated_images: bool = False,
        attacked_images_folder: str = "data/attacks/imagenette_models",
        verbose: bool = True,
    ) -> Dict:
        """
        Train one ImageNette model with progressively growing attacked datasets.

        Each iteration generates new successful adversarial examples for every
        attack in `attack_names` and appends them to the existing training and
        validation attacked sets.
        """
        if model is None:
            raise ValueError("model must be provided")
        if not attack_names:
            raise ValueError("attack_names must be provided")
        if iterations <= 0:
            raise ValueError(f"iterations must be positive, got {iterations}")
        if epochs_per_iteration <= 0:
            raise ValueError(
                f"epochs_per_iteration must be positive, got {epochs_per_iteration}"
            )
        if images_per_attack_per_iteration <= 0:
            raise ValueError(
                "images_per_attack_per_iteration must be positive"
            )

        available_attacks = AttackNames().all_attack_names
        for attack_name in attack_names:
            if attack_name not in available_attacks:
                raise ValueError(
                    f"Attack '{attack_name}' not available. "
                    f"Available attacks: {', '.join(available_attacks)}"
                )

        validation_images_per_attack_per_iteration = (
            images_per_attack_per_iteration
            if validation_images_per_attack_per_iteration is None
            else validation_images_per_attack_per_iteration
        )

        model_name = model.__class__.__name__
        progressive_models_dir = getattr(
            self,
            "progressive_adversarial_models_dir",
            "./models/imagenette_adversarial_progressive",
        )
        os.makedirs(progressive_models_dir, exist_ok=True)

        if save_model_path is None:
            training_day = datetime.now().strftime("%Y%m%d")
            save_model_path = os.path.join(
                progressive_models_dir,
                f"{model_name}_adv_progressive_{training_day}.pt",
            )

        if writer is None:
            run_stamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
            log_dir = (
                f"runs/adversarial_training_progressive/"
                f"{model_name}/{run_stamp}_lr={learning_rate}"
            )
            writer = SummaryWriter(log_dir=log_dir)
            if verbose:
                print(f"📊 TensorBoard logging to: {log_dir}")

        print(f"\n{'=' * 70}")
        print(f"🛡️ Progressive Adversarial Training: {model_name}")
        print(f"{'=' * 70}")
        print("📊 Progressive Training Configuration:")
        print(f"   Attacks: {', '.join(attack_names)}")
        print(f"   Iterations: {iterations}")
        print(f"   Epochs per iteration: {epochs_per_iteration}")
        print(f"   Train images per attack per iteration: {images_per_attack_per_iteration}")
        print(
            f"   Validation images per attack per iteration: "
            f"{validation_images_per_attack_per_iteration}"
        )
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Save generated images: {save_generated_images}")

        # Keep attack generation anchored to the clean dataset on every iteration.
        generation_train_loader, generation_test_loader = load_imagenette(
            batch_size=1,
            shuffle=True,
        )
        clean_train_loader, clean_test_loader = load_imagenette(
            batch_size=batch_size,
            shuffle=False,
        )
        progressive_train_dataset: List[Tuple[torch.Tensor, int]] = []
        progressive_test_dataset: List[Tuple[torch.Tensor, int]] = []

        def iteration_data_generator(
            current_model: torch.nn.Module,
            iteration: int,
        ) -> Dict:
            print(f"\n{'=' * 60}")
            print(f"🔄 Iteration {iteration + 1}/{iterations}")
            print(f"{'=' * 60}")

            new_train_examples, train_distance_score = self._collect_successful_adversarial_examples(
                model=current_model,
                attack_names=attack_names,
                data_loader=generation_train_loader,
                images_per_attack=images_per_attack_per_iteration,
                attack_test_dataset=False,
                save_generated_images=save_generated_images,
                attacked_images_folder=attacked_images_folder,
                iteration=iteration,
                verbose=verbose,
            )
            new_test_examples, _ = self._collect_successful_adversarial_examples(
                model=current_model,
                attack_names=attack_names,
                data_loader=generation_test_loader,
                images_per_attack=validation_images_per_attack_per_iteration,
                attack_test_dataset=True,
                save_generated_images=save_generated_images,
                attacked_images_folder=attacked_images_folder,
                iteration=iteration,
                verbose=verbose,
            )

            if not new_train_examples:
                raise RuntimeError(
                    "No successful adversarial examples were generated for training."
                )

            writer.add_scalar(
                "Distance/train_generated/l0_pixels",
                train_distance_score.l0_pixels,
                iteration + 1,
            )
            writer.add_scalar(
                "Distance/train_generated/l1",
                train_distance_score.l1,
                iteration + 1,
            )
            writer.add_scalar(
                "Distance/train_generated/l2",
                train_distance_score.l2,
                iteration + 1,
            )
            writer.add_scalar(
                "Distance/train_generated/linf",
                train_distance_score.linf,
                iteration + 1,
            )
            writer.add_scalar(
                "Distance/train_generated/power_mse",
                train_distance_score.power_mse,
                iteration + 1,
            )

            progressive_train_dataset.extend(new_train_examples)
            progressive_test_dataset.extend(new_test_examples)


            combined_train_dataset = ConcatDataset(
                [clean_train_loader.dataset, progressive_train_dataset]
            )
            combined_val_dataset = ConcatDataset(
                [clean_test_loader.dataset, progressive_test_dataset]
            )

            writer.add_scalar(
                "Dataset/combined_train_size_iteration",
                len(combined_train_dataset),
                iteration + 1,
            )

            print(
                f"📦 Dataset sizes: "
                f"progressive_train={len(progressive_train_dataset)}, "
                f"progressive_test={len(progressive_test_dataset)}, "
                f"combined_train={len(combined_train_dataset)}, "
                f"combined_test={len(combined_val_dataset)}"
            )

            return {
                "train_loader": DataLoader(
                    combined_train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                ),
                "val_loader": DataLoader(
                    combined_val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                ),
                "adv_test_loader": (
                    DataLoader(
                        progressive_test_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                    )
                    if progressive_test_dataset
                    else None
                ),
                "train_dataset_size": len(combined_train_dataset),
                "val_dataset_size": len(combined_val_dataset),
                "adv_test_dataset_size": len(progressive_test_dataset),
            }

        training_results = Training.train_imagenette_adversarial_progressive(
            model=model,
            iteration_data_generator=iteration_data_generator,
            clean_test_loader=clean_test_loader,
            learning_rate=learning_rate,
            iterations=iterations,
            epochs_per_iteration=epochs_per_iteration,
            device=str(self.device),
            save_model_path=save_model_path,
            writer=writer,
            early_stopping_patience=early_stopping_patience,
            min_delta=0.001,
            scheduler_type=scheduler_type,
            scheduler_params=scheduler_params,
            gradient_clip_norm=gradient_clip_norm,
            weight_decay=weight_decay,
            verbose=verbose,
        )

        total_time = training_results["total_training_time"]

        print(f"\n✅ Progressive adversarial training completed!")
        print(f"   Training time: {total_time:.2f}s ({total_time / 60:.2f} minutes)")
        print(
            f"   Best combined validation accuracy: "
            f"{training_results['best_val_accuracy']:.2f}%"
        )
        print(
            f"   Best adversarial validation accuracy: "
            f"{training_results['best_adv_val_accuracy']:.2f}%"
        )
        print(f"   Model saved to: {save_model_path}")

        return {
            "model_name": model_name,
            "task": "progressive_adversarial_training",
            "attack_names": attack_names,
            "iterations": iterations,
            "epochs_per_iteration": epochs_per_iteration,
            "images_per_attack_per_iteration": images_per_attack_per_iteration,
            "save_generated_images": save_generated_images,
            "attacked_images_folder": attacked_images_folder,
            "training_time": total_time,
            "save_path": save_model_path,
            "best_val_accuracy": training_results["best_val_accuracy"],
            "best_adv_val_accuracy": training_results["best_adv_val_accuracy"],
            "training_results": training_results,
            "success": True,
        }

    def train_multiple_progressive_adversarial_models(
        self,
        models: List[torch.nn.Module],
        attack_names: List[str],
        learning_rate: float = 0.001,
        iterations: int = 5,
        epochs_per_iteration: int = 2,
        batch_size: int = 128,
        images_per_attack_per_iteration: int = 10,
        validation_images_per_attack_per_iteration: Optional[int] = None,
        save_generated_images: bool = False,
        attacked_images_folder: str = "data/attacks/imagenette_models",
    ) -> List[Dict]:
        print(f"\n{'=' * 70}")
        print("🚀 Training Multiple Progressive Adversarial Models")
        print(f"{'=' * 70}")
        print(f"   Models: {len(models)}")
        print(f"   Attacks: {', '.join(attack_names)}")
        print(f"   Iterations: {iterations}")
        print(f"   Epochs per iteration: {epochs_per_iteration}")

        results = []

        for i, model in enumerate(models, 1):
            model_name = model.__class__.__name__
            print(f"\n📊 Model {i}/{len(models)}: {model_name}")

            result = self.train_progressive_adversarial_model(
                model=model,
                attack_names=attack_names,
                learning_rate=learning_rate,
                iterations=iterations,
                epochs_per_iteration=epochs_per_iteration,
                batch_size=batch_size,
                images_per_attack_per_iteration=images_per_attack_per_iteration,
                validation_images_per_attack_per_iteration=(
                    validation_images_per_attack_per_iteration
                ),
                save_generated_images=save_generated_images,
                attacked_images_folder=attacked_images_folder,
                verbose=True,
            )
            results.append(result)

            if result["success"]:
                print(
                    f"✅ {model_name}: "
                    f"Clean Val Acc={result['best_val_accuracy']:.2f}% | "
                    f"Adv Val Acc={result['best_adv_val_accuracy']:.2f}%"
                )
            else:
                print(f"❌ {model_name}: Failed")

        return results


if __name__ == "__main__":
    model_names = [
        # ModelNames().resnet18,
        # ModelNames().densenet121,
        # ModelNames().mobilenet_v2,
        # ModelNames().efficientnet_b0,
        ModelNames().vgg16,
    ]

    print(f"Model names: {model_names}")
    attack_names = AttackNames().all_attack_names

    trainer = ImageNetteAdversarialProgressiveTrainer(device="auto")
    models = []

    for model_name in model_names:
        print(f"\n🔧 Preparing model: {model_name}")
        model = ImageNetModels.get_model(model_name)
        model.__class__.__name__ = model_name
        model = SetupPretraining.setup_imagenette(model)
        models.append(model)

    trainer.train_multiple_progressive_adversarial_models(
        models=models,
        attack_names=attack_names,
        learning_rate=0.001,
        iterations=30,
        epochs_per_iteration=50,
        batch_size=32,
        images_per_attack_per_iteration=5,
        validation_images_per_attack_per_iteration=1,
        save_generated_images=True,
        attacked_images_folder="data/attacks/imagenette_models",
    )
