#!/usr/bin/env python3
"""
Training Lab - Simple script for testing the robust train_imagenette method.
Similar to pytorch_lab.py - just run the file to test training functionality.
"""

import torch
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.imagenet_models import ImageNetModels
from training.train import Training
from domain.model_names import ModelNames
from data_eng.dataset_loader import load_imagenette

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

# Available models
available_models = [
    ModelNames.resnet18,
    ModelNames.densenet121,
    ModelNames.vgg16,
    ModelNames.mobilenet_v2,
    ModelNames.efficientnet_b0
]

# Basic training configurations
training_configs = {
    'quick_test': {
        'num_epochs': 2,
        'learning_rate': 0.001,
        'batch_size': 16,
        'verbose': True
    },
    'standard': {
        'num_epochs': 10,
        'learning_rate': 0.001,
        'batch_size': 32,
        'scheduler_type': 'step',
        'scheduler_params': {'step_size': 3, 'gamma': 0.8},
        'weight_decay': 1e-4,
        'verbose': True
    },
    'advanced': {
        'num_epochs': 20,
        'learning_rate': 0.001,
        'batch_size': 64,
        'scheduler_type': 'plateau',
        'scheduler_params': {'factor': 0.5, 'patience': 3},
        'weight_decay': 1e-4,
        'gradient_clip_norm': 1.0,
        'verbose': True
    }
}

def load_real_imagenette_dataset(batch_size=16, train_subset_size=-1, test_subset_size=-1):
    """Load real ImageNette dataset."""
    print(f"📊 Loading real ImageNette dataset...")
    print(f"   Batch size: {batch_size}")
    if train_subset_size > 0:
        print(f"   Train subset size: {train_subset_size}")
    if test_subset_size > 0:
        print(f"   Test subset size: {test_subset_size}")
    
    try:
        train_loader, test_loader = load_imagenette(
            transform=None,  # Uses imagenette_transformer() by default
            path_to_data='./data/imagenette',
            batch_size=batch_size,
            train_subset_size=train_subset_size,
            test_subset_size=test_subset_size,
            shuffle=True
        )
        
        print(f"   Train samples: {len(train_loader.dataset)}")
        print(f"   Test samples: {len(test_loader.dataset)}")
        print(f"   ✅ Real ImageNette dataset loaded successfully!")
        
        return train_loader, test_loader
        
    except Exception as e:
        print(f"   ❌ Failed to load real ImageNette dataset: {str(e)}")
        print(f"   📝 Make sure ImageNette dataset is available at './data/imagenette/'")
        raise RuntimeError(f"Cannot load ImageNette dataset: {str(e)}")


def train_single_model(model_name, config_name='quick_test'):
    """Train a single model with specified configuration."""
    print(f"\n{'='*60}")
    print(f"🧪 Training {model_name} with {config_name} configuration")
    print(f"{'='*60}")
    
    try:
        # Load model
        print(f"📥 Loading {model_name}...")
        model = ImageNetModels.get_model(model_name)
        
        # Create dataset
        config = training_configs[config_name]
        train_loader, val_loader = load_real_imagenette_dataset(
            batch_size=config['batch_size'],
            train_subset_size=1000,  # Use subset for faster training
            test_subset_size=200
        )
        
        # Setup save path
        save_path = f"./models/imagenette/{model_name}_{config_name}.pt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Train model
        print(f"🏋️ Starting training...")
        start_time = datetime.now()
        
        training_results = Training.train_imagenette(
            model=model,
            train_loader=train_loader,
            test_loader=val_loader,
            learning_rate=config['learning_rate'],
            num_epochs=config['num_epochs'],
            device=device,
            save_model_path=save_path,
            model_name=model_name,
            writer=None,  # Disable TensorBoard for testing
            setup_model=True,
            scheduler_type=config.get('scheduler_type', 'step'),
            scheduler_params=config.get('scheduler_params', None),
            gradient_clip_norm=config.get('gradient_clip_norm', None),
            weight_decay=config.get('weight_decay', 0.0),
            verbose=config.get('verbose', True)
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Print results summary
        print(f"\n✅ Training completed successfully!")
        print(f"   Training time: {training_time:.2f}s")
        print(f"   Best validation accuracy: {training_results['best_val_accuracy']:.2f}%")
        print(f"   Final training loss: {training_results['train_losses'][-1]:.4f}")
        
        return {
            'success': True,
            'model_name': model_name,
            'config': config_name,
            'training_time': training_time,
            'best_val_accuracy': training_results['best_val_accuracy'],
            'final_train_loss': training_results['train_losses'][-1]
        }
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return {
            'success': False,
            'model_name': model_name,
            'config': config_name,
            'error': str(e)
        }

def train_multiple_models(model_names, config_name='quick_test'):
    """Train multiple models with the same configuration."""
    print(f"\n{'='*60}")
    print(f"🚀 Training Multiple Models with {config_name} configuration")
    print(f"{'='*60}")
    
    results = []
    
    for model_name in model_names:
        result = train_single_model(model_name, config_name)
        results.append(result)
        
        if result['success']:
            print(f"✅ {model_name}: {result['best_val_accuracy']:.2f}% accuracy")
        else:
            print(f"❌ {model_name}: Failed - {result['error']}")
    
    return results

# Example usage - uncomment what you want to test:

train_single_model(ModelNames.resnet18, 'quick_test')

# Multiple models training
# train_multiple_models([ModelNames.resnet18, ModelNames.densenet121], 'quick_test')

# All models training
# train_multiple_models(available_models, 'quick_test')

# Different configurations
# train_single_model(ModelNames.resnet18, 'standard')
# train_single_model(ModelNames.resnet18, 'advanced')

