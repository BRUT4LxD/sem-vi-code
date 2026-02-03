#!/usr/bin/env python3
"""
ImageNette Model Trainer - facade for standard, noise-detection, and adversarial training.
"""

from imagenette_lab.imagenette_adversarial_trainer import ImageNetteAdversarialTrainer
from imagenette_lab.imagenette_noise_detection_trainer import ImageNetteNoiseDetectionTrainer
from imagenette_lab.imagenette_standard_trainer import ImageNetteStandardTrainer


class ImageNetteModelTrainer(
    ImageNetteStandardTrainer,
    ImageNetteNoiseDetectionTrainer,
    ImageNetteAdversarialTrainer,
):
    """
    Convenience trainer that combines standard, noise detection,
    and adversarial training interfaces.
    """

    def __init__(self, device: str = 'auto', models_dir: str = './models/imagenette'):
        super().__init__(device=device, models_dir=models_dir)