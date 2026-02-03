import os
import sys
from typing import Dict, List

import torch

from config.imagenette_classes import ImageNetteClasses
from imagenette_training_configs import ImageNetteTrainingConfigs

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BaseImageNetteTrainer:
    """
    Base trainer with shared setup and helper methods for ImageNette training.
    """

    def __init__(self, device: str = 'auto', models_dir: str = './models/imagenette'):
        """
        Initialize the ImageNette trainer base.

        Args:
            device: Device to use ('auto', 'cuda', or 'cpu')
            models_dir: Directory to save trained models
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.models_dir = models_dir
        self.noise_detection_dir = './models/noise_detection'
        self.adversarial_models_dir = './models/imagenette_adversarial'
        self.AVAILABLE_MODELS = ImageNetteTrainingConfigs.AVAILABLE_MODELS
        self.AVAILABLE_CONFIGS = ImageNetteTrainingConfigs.ALL_CONFIGS

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.noise_detection_dir, exist_ok=True)
        os.makedirs(self.adversarial_models_dir, exist_ok=True)

        print(f"🚀 ImageNette trainer initialized on device: {self.device}")
        print(f"📁 Models will be saved to: {self.models_dir}")
        print(f"📁 Noise detection models will be saved to: {self.noise_detection_dir}")
        print(f"📁 Adversarial models will be saved to: {self.adversarial_models_dir}")
        print("🎯 Dataset: ImageNette (10 classes)")

    def list_available_configs(self) -> Dict[str, Dict]:
        """List all available ImageNette training configurations."""
        print("\n📋 Available ImageNette Training Configurations:")
        print(f"{'='*70}")
        ImageNetteTrainingConfigs.print_config_summary()
        return ImageNetteTrainingConfigs.list_configs()

    def list_available_models(self) -> List[str]:
        """List all available models for ImageNette training."""
        print("\n🤖 Available Models for ImageNette Training:")
        print(f"{'='*70}")
        for i, model_name in enumerate(self.AVAILABLE_MODELS, 1):
            print(f"   {i}. {model_name}")

        return self.AVAILABLE_MODELS

    def list_imagenette_classes(self) -> List[str]:
        """List all ImageNette classes."""
        print("\n🎯 ImageNette Classes (10 classes):")
        print(f"{'='*70}")
        classes = ImageNetteClasses.get_classes()
        for i, class_name in enumerate(classes, 1):
            print(f"   {i}. {class_name}")

        return classes
