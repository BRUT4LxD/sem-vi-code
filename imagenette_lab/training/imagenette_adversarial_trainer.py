import os
from datetime import datetime
from typing import Dict, List

import torch

from attacks.attack_names import AttackNames
from data_eng.dataset_loader import load_attacked_imagenette_for_adversarial_training, load_imagenette
from imagenette_lab.imagenette_base_trainer import BaseImageNetteTrainer
from training.train import Training


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
        attacked_images_folder: str = "data/attacks/imagenette_models",
        clean_images_folder: str = "./data/imagenette/train",
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
            attacked_images_folder: Folder with pre-attacked images (used if use_preattacked_images=True)
            clean_images_folder: Folder with clean images
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
                # Load pre-attacked images mixed with clean images
                train_loader = load_attacked_imagenette_for_adversarial_training(
                    attacked_images_folder=attacked_images_folder,
                    clean_images_folder=clean_images_folder,
                    batch_size=batch_size,
                    shuffle=True,
                    train_dataset=True
                )
                # Load test dataset with pre-attacked images
                test_loader = load_attacked_imagenette_for_adversarial_training(
                    attacked_images_folder=attacked_images_folder,
                    clean_images_folder="./data/imagenette/val",
                    batch_size=batch_size,
                    shuffle=False,
                    train_dataset=False
                )
            else:
                # Load clean ImageNette for on-the-fly generation
                train_loader, test_loader = load_imagenette(
                    batch_size=batch_size,
                    shuffle=True
                )

            # Setup save path (distinguish between pre-attacked and on-the-fly modes)
            mode_suffix = "preattacked" if use_preattacked_images else "onthefly"
            save_path = os.path.join(self.adversarial_models_dir, f"{model_name}_adv_{mode_suffix}.pt")

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
        attacked_images_folder: str = "data/attacks/imagenette_models"
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
            attacked_images_folder: Folder with pre-attacked images

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
                attacked_images_folder=attacked_images_folder,
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
