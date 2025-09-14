#!/usr/bin/env python3
"""
Training Configurations - Reusable training configurations for ImageNette models.

This module provides standardized training configurations that can be used across
different training modules and scripts.

Usage:
    from training.training_configs import ImageNetteTrainingConfigs
    
    config = ImageNetteTrainingConfigs.get_config(ImageNetteTrainingConfigs.TEST)
    # or
    config = ImageNetteTrainingConfigs.TEST_CONFIG
"""

from typing import Dict

from domain.model.model_names import ModelNames


class ImageNetteTrainingConfigs:
    """
    Standardized training configurations for ImageNette models.
    
    All configurations are optimized for the ImageNette dataset (10 classes)
    and include appropriate hyperparameters for different training scenarios.
    """
    
    # Configuration names as constants
    TEST = 'test'
    STANDARD = 'standard'
    ADVANCED = 'advanced'
    
    # Test configuration for rapid prototyping
    TEST_CONFIG = {
        'num_epochs': 2,
        'learning_rate': 0.001,
        'batch_size': 16,
        'train_subset_size': 2000,
        'test_subset_size': 1000,
        'verbose': True,
        'description': 'Quick test configuration for rapid prototyping on ImageNette'
    }
    
    # Standard configuration for balanced training
    STANDARD_CONFIG = {
        'num_epochs': 10,
        'learning_rate': 0.001,
        'batch_size': 32,
        'train_subset_size': 5000,
        'test_subset_size': 2000,
        'scheduler_type': 'step',
        'scheduler_params': {'step_size': 3, 'gamma': 0.8},
        'weight_decay': 1e-4,
        'verbose': True,
        'description': 'Standard configuration for balanced ImageNette training'
    }
    
    # Advanced configuration for maximum performance
    ADVANCED_CONFIG = {
        'num_epochs': 30,
        'learning_rate': 0.001,
        'batch_size': 64,
        'train_subset_size': -1,  # Use full ImageNette dataset
        'test_subset_size': -1,   # Use full ImageNette dataset
        'scheduler_type': 'cosine',
        'scheduler_params': {'eta_min': 1e-6},
        'weight_decay': 1e-4,
        'gradient_clip_norm': 1.0,
        'early_stopping_patience': 5,
        'verbose': True,
        'description': 'Full ImageNette dataset configuration for maximum performance'
    }
    
    # All available configurations
    ALL_CONFIGS = {
        TEST: TEST_CONFIG,
        STANDARD: STANDARD_CONFIG,
        ADVANCED: ADVANCED_CONFIG
    }
    
    # Available models for ImageNette training
    AVAILABLE_MODELS = [
        ModelNames.resnet18,
        ModelNames.resnet50,
        ModelNames.resnet101,
        ModelNames.resnet152,
        ModelNames.densenet121,
        ModelNames.densenet161,
        ModelNames.densenet169,
        ModelNames.densenet201,
        ModelNames.vgg11,
        ModelNames.vgg13,
        ModelNames.vgg16,
        ModelNames.vgg19,
        ModelNames.mobilenet_v2,
        ModelNames.efficientnet_b0
    ]
    
    @classmethod
    def get_config(cls, config_name: str) -> Dict:
        """
        Get a training configuration by name.
        
        Args:
            config_name: Name of the configuration
            
        Returns:
            dict: Training configuration
            
        Raises:
            ValueError: If configuration name is not found
        """
        if config_name not in cls.ALL_CONFIGS:
            available = ', '.join(cls.ALL_CONFIGS.keys())
            raise ValueError(f"Configuration '{config_name}' not found. Available: {available}")
        
        return cls.ALL_CONFIGS[config_name].copy()
    
    @classmethod
    def list_configs(cls) -> Dict[str, Dict]:
        """
        List all available training configurations.
        
        Returns:
            dict: All available configurations
        """
        return cls.ALL_CONFIGS.copy()
    
    @classmethod
    def print_config_summary(cls):
        """
        Print a summary of all training configurations.
        """
        print(f"\n📋 Available ImageNette Training Configurations:")
        print(f"{'='*70}")
        for name in cls.ALL_CONFIGS.keys():
            config = cls.ALL_CONFIGS[name]
            print(f"\n🔧 {name.upper()}:")
            print(f"   Description: {config['description']}")
            print(f"   Epochs: {config['num_epochs']}")
            print(f"   Learning Rate: {config['learning_rate']}")
            print(f"   Batch Size: {config['batch_size']}")
            print(f"   Train Samples: {config.get('train_subset_size', 'Full ImageNette dataset')}")
            print(f"   Test Samples: {config.get('test_subset_size', 'Full ImageNette dataset')}")
            if 'scheduler_type' in config:
                print(f"   Scheduler: {config['scheduler_type']}")
            if 'weight_decay' in config:
                print(f"   Weight Decay: {config['weight_decay']}")


# Convenience functions for quick access
def get_config(config_name: str) -> Dict:
    """Quick access to get a configuration."""
    return ImageNetteTrainingConfigs.get_config(config_name)

def list_configs() -> Dict[str, Dict]:
    """Quick access to list all configurations."""
    return ImageNetteTrainingConfigs.list_configs()

def print_configs():
    """Quick access to print configuration summary."""
    ImageNetteTrainingConfigs.print_config_summary()


if __name__ == "__main__":
    # Example usage
    print("ImageNette Training Configurations")
    print("=" * 50)
    
    # Show all configurations
    ImageNetteTrainingConfigs.print_config_summary()
    
    # Get configuration using constants
    config = ImageNetteTrainingConfigs.get_config(ImageNetteTrainingConfigs.TEST)
    print(f"\nTest config epochs: {config['num_epochs']}")
    
    # Get configuration using string
    config = ImageNetteTrainingConfigs.get_config('standard')
    print(f"Standard config epochs: {config['num_epochs']}")