import os
from datetime import datetime
from typing import Dict, List, Optional

import torch

from attacks.attack_names import AttackNames
from data_eng.adversarial_training_dataset_builder import (
    build_imagenette_adversarial_training_loaders,
)
from config.imagenet_models import ImageNetModels
from data_eng.dataset_loader import load_imagenette
from domain.model.model_names import ModelNames
from imagenette_lab.training.imagenette_base_trainer import BaseImageNetteTrainer
from training.train import Training
from training.transfer.setup_pretraining import SetupPretraining


class ImageNetteAdversarialTrainer(BaseImageNetteTrainer):
    """
    Adversarial training utilities for ImageNette models.
    """

    def train_adversarial_model(
        self,
        model: torch.nn.Module = None,
        attack_names: List[str] = None,
        learning_rate: float = 0.001,
        num_epochs: int = 20,
        batch_size: int = 32,
        adversarial_ratio: float = 0.5,
        early_stopping_patience: int = 7,
        scheduler_type: str = 'step',
        weight_decay: float = 0.0001,
        gradient_clip_norm: float = 1.0,
        use_preattacked_images: bool = False,
        attacked_subset_size: int = -1,
        augment_clean_to_match_attacked: bool = True,
        train_test_split: Optional[float] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Train a model using adversarial training on ImageNette dataset.

        Supports two modes:
        1. On-the-fly: Generate adversarial examples during training
        2. Pre-attacked: Use pre-generated adversarial images from disk

        Args:
            model: Model to train
            attack_names: List of attacks for on-the-fly mode (required if use_preattacked_images=False)
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            adversarial_ratio: Ratio of adversarial to clean examples for on-the-fly mode (0.5 = 50/50)
            early_stopping_patience: Epochs to wait before early stopping
            scheduler_type: Learning rate scheduler type ('step', 'plateau', 'cosine')
            weight_decay: L2 regularization weight
            gradient_clip_norm: Gradient clipping threshold
            use_preattacked_images: If True, use pre-attacked images from disk
            attacked_subset_size: Optional cap for attacked train/test images (-1 uses all)
            augment_clean_to_match_attacked: If True, oversample clean images to match attacked count
            train_test_split: Optional ratio to resplit attacked+clean data (e.g., 0.8)
            verbose: Whether to print detailed progress

        Returns:
            dict: Training results including model, metrics, and metadata
        """
        model_name = model.__class__.__name__

        print(f"\n{'='*70}")
        print(f"🛡️ Adversarial Training: {model_name}")
        print(f"{'='*70}")

        try:
            # Validate attack names (only for on-the-fly mode)
            if not use_preattacked_images:
                if attack_names is None or len(attack_names) == 0:
                    raise ValueError("attack_names must be provided for on-the-fly adversarial training")

                available_attacks = AttackNames().all_attack_names
                for attack_name in attack_names:
                    if attack_name not in available_attacks:
                        raise ValueError(
                            f"Attack '{attack_name}' not available. Available attacks: {', '.join(available_attacks)}"
                        )

            print("📊 Adversarial Training Configuration:")
            print(f"   Model: {model_name}")
            print(f"   Mode: {'Pre-attacked Images' if use_preattacked_images else 'On-the-fly Generation'}")
            if not use_preattacked_images:
                print(f"   Attacks: {', '.join(attack_names)}")
                print(f"   Adversarial ratio: {adversarial_ratio:.1%}")
            print(f"   Learning Rate: {learning_rate}")
            print(f"   Epochs: {num_epochs}")
            print(f"   Batch size: {batch_size}")

            # Load dataset based on mode
            print("\n📁 Loading dataset...")
            if use_preattacked_images:
                train_loader, test_loader = build_imagenette_adversarial_training_loaders(
                    batch_size=batch_size,
                    attacked_subset_size=attacked_subset_size,
                    clean_to_attacked_ratio=1.0,
                    augment_clean_to_match_attacked=augment_clean_to_match_attacked,
                    train_test_split=train_test_split,
                )
            else:
                # Load clean ImageNette for on-the-fly generation
                train_loader, test_loader = load_imagenette(
                    batch_size=batch_size,
                    shuffle=True
                )

            # Setup save path (distinguish between pre-attacked and on-the-fly modes)
            mode_suffix = "preattacked" if use_preattacked_images else "onthefly"
            training_day = datetime.now().strftime("%Y%m%d")
            save_path = os.path.join(
                self.adversarial_models_dir,
                f"{model_name}_adv_{mode_suffix}_{training_day}.pt",
            )

            # Train model using modern adversarial training
            print("\n🔥 Starting adversarial training...")
            start_time = datetime.now()

            training_results = Training.train_imagenette_adversarial(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                attack_names=attack_names,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                device=str(self.device),
                save_model_path=save_path,
                writer=None,  # Will be created automatically
                validation_frequency=1,
                early_stopping_patience=early_stopping_patience,
                min_delta=0.001,
                scheduler_type=scheduler_type,
                scheduler_params=None,
                gradient_clip_norm=gradient_clip_norm,
                weight_decay=weight_decay,
                adversarial_ratio=adversarial_ratio,
                use_preattacked_images=use_preattacked_images,
                verbose=verbose
            )

            training_time = (datetime.now() - start_time).total_seconds()

            # Print results summary
            print("\n✅ Adversarial training completed successfully!")
            print(f"   Training time: {training_time:.2f}s ({training_time/60:.2f} minutes)")
            print(f"   Best validation accuracy: {training_results['best_val_accuracy']:.2f}%")
            print(f"   Model saved to: {save_path}")

            return {
                'model_name': model_name,
                'task': 'adversarial_training',
                'attack_names': attack_names,
                'use_preattacked_images': use_preattacked_images,
                'num_epochs': num_epochs,
                'learning_rate': learning_rate,
                'training_time': training_time,
                'save_path': save_path,
                'best_val_accuracy': training_results['best_val_accuracy'],
                'training_results': training_results,
                'success': True
            }

        except Exception as e:
            error_msg = f"Adversarial training failed: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                'model_name': model_name,
                'task': 'adversarial_training',
                'error': error_msg,
                'success': False
            }

    def train_multiple_adversarial_models(
        self,
        models: List[torch.nn.Module],
        attack_names: List[str] = None,
        learning_rate: float = 0.001,
        num_epochs: int = 20,
        batch_size: int = 32,
        adversarial_ratio: float = 0.5,
        use_preattacked_images: bool = False,
        attacked_subset_size: int = -1,
        augment_clean_to_match_attacked: bool = True,
        train_test_split: Optional[float] = None,
    ) -> List[Dict]:
        """
        Train multiple models using adversarial training.

        Args:
            models: List of preloaded ImageNette models to train adversarially
            attack_names: List of attacks for on-the-fly mode (required if use_preattacked_images=False)
            learning_rate: Learning rate for all models
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            adversarial_ratio: Ratio of adversarial to clean examples for on-the-fly mode
            use_preattacked_images: If True, use pre-attacked images from disk
            attacked_subset_size: Optional cap for attacked train/test images (-1 uses all)
            augment_clean_to_match_attacked: If True, oversample clean images to match attacked count
            train_test_split: Optional ratio to resplit attacked+clean data (e.g., 0.8)

        Returns:
            List of training results for each model
        """
        print(f"\n{'='*70}")
        print("🚀 Training Multiple Adversarial Models")
        print(f"{'='*70}")
        print(f"   Models: {len(models)}")
        print(f"   Mode: {'Pre-attacked Images' if use_preattacked_images else 'On-the-fly Generation'}")
        if not use_preattacked_images:
            print(f"   Attacks: {', '.join(attack_names)}")
            print(f"   Adversarial Ratio: {adversarial_ratio:.1%}")
        print(f"   Epochs: {num_epochs}")

        results = []

        for i, model in enumerate(models, 1):
            model_name = model.__class__.__name__
            print(f"\n📊 Model {i}/{len(models)}: {model_name}")

            result = self.train_adversarial_model(
                model=model,
                attack_names=attack_names,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                batch_size=batch_size,
                adversarial_ratio=adversarial_ratio,
                use_preattacked_images=use_preattacked_images,
                attacked_subset_size=attacked_subset_size,
                augment_clean_to_match_attacked=augment_clean_to_match_attacked,
                train_test_split=train_test_split,
                verbose=True
            )

            results.append(result)

            if result['success']:
                print(f"✅ {model_name}: Val Acc={result['best_val_accuracy']:.2f}%")
            else:
                print(f"❌ {model_name}: Failed - {result['error']}")

        # Summary
        print("\n📈 Adversarial Training Summary:")
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]

        print(f"   Successful: {len(successful)}/{len(results)}")
        print(f"   Failed: {len(failed)}/{len(results)}")

        if successful:
            print("\n🏆 Best Adversarially Trained Models (by validation accuracy):")
            successful.sort(key=lambda x: x['best_val_accuracy'], reverse=True)
            for i, result in enumerate(successful[:5], 1):
                print(f"   {i}. {result['model_name']}: "
                      f"Val Acc={result['best_val_accuracy']:.2f}%")

        return results
if __name__ == "__main__":
    model_names = [
        ModelNames().resnet18,
        ModelNames().densenet121,
        ModelNames().mobilenet_v2,
        ModelNames().efficientnet_b0,
        ModelNames().vgg16,
    ]

    attack_names = AttackNames().all_attack_names

    use_preattacked_images = True
    attacked_subset_size = 20000
    augment_clean_to_match_attacked = True
    train_test_split = 0.2
    learning_rate = 0.001
    num_epochs = 1000
    batch_size = 128
    adversarial_ratio = 0.5
    device = "auto"

    print("🚀 Starting ImageNette adversarial training experiment...")
    print(f"📊 Models: {model_names}")
    print(f"🛡️ Mode: {'Pre-attacked Images' if use_preattacked_images else 'On-the-fly Generation'}")
    if not use_preattacked_images:
        print(f"🎯 Attacks: {attack_names}")

    trainer = ImageNetteAdversarialTrainer(device=device)
    models = []

    for model_name in model_names:
        print(f"\n🔧 Preparing model: {model_name}")
        model = ImageNetModels.get_model(model_name)
        model.__class__.__name__ = model_name
        model = SetupPretraining.setup_imagenette(model, full_finetune=True)
        models.append(model)

    trainer.train_multiple_adversarial_models(
        models=models,
        attack_names=attack_names,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        adversarial_ratio=adversarial_ratio,
        use_preattacked_images=use_preattacked_images,
        attacked_subset_size=attacked_subset_size,
        augment_clean_to_match_attacked=augment_clean_to_match_attacked,
        train_test_split=train_test_split,
    )
