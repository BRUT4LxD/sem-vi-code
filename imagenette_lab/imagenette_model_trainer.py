#!/usr/bin/env python3
"""
ImageNette Model Trainer - Complete training and validation system for ImageNette models.

This module provides a comprehensive training system specifically designed for ImageNette
(10-class subset of ImageNet). It includes all necessary components for training, 
validation, and model management.

Usage:
    from training.imagenette_model_trainer import ImageNetteModelTrainer
    from training.training_configs import ImageNetteTrainingConfigs
    
    trainer = ImageNetteModelTrainer()
    trainer.train_model('resnet18', ImageNetteTrainingConfigs.ADVANCED)
    trainer.validate_model('./models/imagenette/resnet18_advanced.pt')
"""

import torch
import os
import sys
from datetime import datetime
from typing import List, Dict, Optional, Literal

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.imagenet_models import ImageNetModels
from config.imagenette_classes import ImageNetteClasses
from domain.model.model_names import ModelNames
from data_eng.dataset_loader import load_imagenette
from data_eng.io import load_model_imagenette
from training.train import Training
from training.transfer.setup_pretraining import SetupPretraining
from imagenette_training_configs import ImageNetteTrainingConfigs
from evaluation.metrics import Metrics
from evaluation.validation import Validation
from evaluation.visualization import simple_visualize


class ImageNetteModelTrainer:
    """
    Comprehensive ImageNette model training and validation system.
    
    This class provides all necessary functionality to train ImageNette models from scratch,
    validate them, and manage the training process. Specifically designed for the 10-class
    ImageNette dataset.
    """
    
    def __init__(self, device: str = 'auto', models_dir: str = './models/imagenette'):
        """
        Initialize the ImageNetteModelTrainer.
        
        Args:
            device: Device to use ('auto', 'cuda', or 'cpu')
            models_dir: Directory to save trained models
        """
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        
        print(f"🚀 ImageNetteModelTrainer initialized on device: {self.device}")
        print(f"📁 Models will be saved to: {self.models_dir}")
        print(f"🎯 Dataset: ImageNette (10 classes)")
    
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
    
    AVAILABLE_CONFIGS = [
        ImageNetteTrainingConfigs.TEST,
        ImageNetteTrainingConfigs.STANDARD,
        ImageNetteTrainingConfigs.ADVANCED
    ]
    
    def list_available_configs(self) -> Dict[str, Dict]:
        """List all available ImageNette training configurations."""
        print(f"\n📋 Available ImageNette Training Configurations:")
        print(f"{'='*70}")
        ImageNetteTrainingConfigs.print_config_summary()
        return ImageNetteTrainingConfigs.list_configs()
    
    def list_available_models(self) -> List[str]:
        """List all available models for ImageNette training."""
        print(f"\n🤖 Available Models for ImageNette Training:")
        print(f"{'='*70}")
        for i, model_name in enumerate(self.AVAILABLE_MODELS, 1):
            print(f"   {i}. {model_name}")
        
        return self.AVAILABLE_MODELS
    
    def list_imagenette_classes(self) -> List[str]:
        """List all ImageNette classes."""
        print(f"\n🎯 ImageNette Classes (10 classes):")
        print(f"{'='*70}")
        classes = ImageNetteClasses.get_classes()
        for i, class_name in enumerate(classes, 1):
            print(f"   {i}. {class_name}")
        
        return classes
    
    def train_model(self, model_name: str, config_name: str = ImageNetteTrainingConfigs.STANDARD, 
                   custom_config: Optional[Dict] = None) -> Dict:
        """
        Train an ImageNette model with specified configuration.
        
        Args:
            model_name: Name of the model to train
            config_name: Name of the training configuration (must be from ImageNetteTrainingConfigs)
            custom_config: Custom configuration to override defaults
            
        Returns:
            dict: Training results including model, metrics, and metadata
            
        Raises:
            ValueError: If model_name or config_name is not valid
        """
        print(f"\n{'='*70}")
        print(f"🏋️ Training {model_name} for ImageNette with {config_name} configuration")
        print(f"{'='*70}")
        
        try:
            # Validate inputs
            if model_name not in self.AVAILABLE_MODELS:
                available_models = ', '.join(self.AVAILABLE_MODELS)
                raise ValueError(f"Model '{model_name}' not available. Available models: {available_models}")
            
            if config_name not in self.AVAILABLE_CONFIGS:
                available_configs = ', '.join(self.AVAILABLE_CONFIGS)
                raise ValueError(f"Configuration '{config_name}' not available. Available configurations: {available_configs}")
            
            # Get configuration from training_configs module
            config = ImageNetteTrainingConfigs.get_config(config_name)
            if custom_config:
                config.update(custom_config)
            
            print(f"📊 ImageNette Training Configuration:")
            print(f"   Model: {model_name}")
            print(f"   Dataset: ImageNette (10 classes)")
            print(f"   Epochs: {config['num_epochs']}")
            print(f"   Learning Rate: {config['learning_rate']}")
            print(f"   Batch Size: {config['batch_size']}")
            print(f"   Train Samples: {config.get('train_subset_size', 'Full ImageNette dataset')}")
            print(f"   Test Samples: {config.get('test_subset_size', 'Full ImageNette dataset')}")
            
            # Create ImageNette dataset
            print(f"\n📥 Loading ImageNette dataset...")
            train_loader, test_loader = load_imagenette(
                batch_size=config['batch_size'],
                train_subset_size=config.get('train_subset_size', -1),
                test_subset_size=config.get('test_subset_size', -1),
            )
            
            # Setup save path
            save_path = os.path.join(self.models_dir, f"{model_name}_{config_name}.pt")
            
            # Train model
            print(f"\n🚀 Starting ImageNette training...")
            start_time = datetime.now()
            
            training_results = Training.train_imagenette(
                model=ImageNetModels.get_model(model_name),
                train_loader=train_loader,
                test_loader=test_loader,
                learning_rate=config['learning_rate'],
                num_epochs=config['num_epochs'],
                device=self.device,
                save_model_path=save_path,
                model_name=model_name,
                writer=None,
                setup_model=True,  # Automatically setup for ImageNette
                scheduler_type=config.get('scheduler_type', 'step'),
                scheduler_params=config.get('scheduler_params', None),
                gradient_clip_norm=config.get('gradient_clip_norm', None),
                weight_decay=config.get('weight_decay', 0.0),
                early_stopping_patience=config.get('early_stopping_patience', 5),
                verbose=config.get('verbose', True)
            )
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Print results summary
            print(f"\n✅ ImageNette training completed successfully!")
            print(f"   Training time: {training_time:.2f}s")
            print(f"   Best validation accuracy: {training_results['best_val_accuracy']:.2f}%")
            print(f"   Final training loss: {training_results['train_losses'][-1]:.4f}")
            print(f"   Model saved to: {save_path}")
            
            return {
                'model_name': model_name,
                'config_name': config_name,
                'config': config,
                'training_results': training_results,
                'training_time': training_time,
                'save_path': save_path,
                'dataset': 'ImageNette',
                'num_classes': 10,
                'success': True
            }
            
        except Exception as e:
            error_msg = f"ImageNette training failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'model_name': model_name,
                'config_name': config_name,
                'error': error_msg,
                'success': False
            }
    
    def train_multiple_models(self, model_names: List[str], config_name: str = ImageNetteTrainingConfigs.STANDARD) -> List[Dict]:
        """
        Train multiple ImageNette models with the same configuration.
        
        Args:
            model_names: List of model names to train
            config_name: Training configuration to use (must be from ImageNetteTrainingConfigs)
            
        Returns:
            List of training results for each model
            
        Raises:
            ValueError: If config_name is not valid
        """
        print(f"\n{'='*70}")
        print(f"🚀 Training Multiple ImageNette Models with {config_name} configuration")
        print(f"{'='*70}")
        
        # Validate configuration
        if config_name not in self.AVAILABLE_CONFIGS:
            available_configs = ', '.join(self.AVAILABLE_CONFIGS)
            raise ValueError(f"Configuration '{config_name}' not available. Available configurations: {available_configs}")
        
        results = []
        
        for i, model_name in enumerate(model_names, 1):
            print(f"\n📊 Model {i}/{len(model_names)}: {model_name}")
            result = self.train_model(model_name, config_name)
            results.append(result)
            
            if result['success']:
                print(f"✅ {model_name}: {result['training_results']['best_val_accuracy']:.2f}% accuracy")
            else:
                print(f"❌ {model_name}: Failed - {result['error']}")
        
        # Summary
        print(f"\n📈 ImageNette Training Summary:")
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"   Successful: {len(successful)}/{len(results)}")
        print(f"   Failed: {len(failed)}/{len(results)}")
        
        if successful:
            print(f"\n🏆 Best ImageNette Performers:")
            successful.sort(key=lambda x: x['training_results']['best_val_accuracy'], reverse=True)
            for i, result in enumerate(successful[:3], 1):
                print(f"   {i}. {result['model_name']}: {result['training_results']['best_val_accuracy']:.2f}%")
        
        return results
    
    def validate_model(self, model_path: str, batch_size: int = 16, 
                      verbose: bool = True, show_visualization: bool = True) -> Dict:
        """
        Validate a trained ImageNette model.
        
        Args:
            model_path: Path to the saved model
            batch_size: Batch size for validation
            verbose: Whether to print detailed results
            show_visualization: Whether to show visual verification
            
        Returns:
            dict: Validation results including metrics and model info
        """
        print(f"\n{'='*70}")
        print(f"📊 Validating ImageNette Model: {model_path}")
        print(f"{'='*70}")
        
        try:
            # Load model
            model_info = load_model_imagenette(model_path, device=str(self.device), verbose=verbose)
            
            if not model_info['success']:
                raise RuntimeError(model_info['error'])
            
            # Create ImageNette test dataset
            _, test_loader = load_imagenette(batch_size=batch_size)
            
            # Evaluate model on ImageNette
            print(f"\n🔍 Running ImageNette evaluation...")
            acc, prec, rec, f1 = Metrics.evaluate_model_torchmetrics(
                model_info['model'], test_loader, 10, verbose=verbose  # 10 classes for ImageNette
            )
            
            # Visual verification with ImageNette classes
            if show_visualization:
                print(f"\n🖼️ Running ImageNette visual verification...")
                imagenette_classes = ImageNetteClasses.get_classes()
                _, viz_test_loader = load_imagenette(batch_size=8, test_subset_size=50)
                
                simple_visualize(
                    test_model=model_info['model'],
                    test_loader=viz_test_loader,
                    batch_size=8,
                    classes=imagenette_classes,
                    device=self.device
                )
            
            # Summary
            print(f"\n🎯 ImageNette Validation Summary:")
            print(f"   Model: {model_info['model_name']}")
            print(f"   Dataset: ImageNette (10 classes)")
            print(f"   Test Accuracy: {acc:.4f}")
            print(f"   Precision: {prec:.4f}")
            print(f"   Recall: {rec:.4f}")
            print(f"   F1-Score: {f1:.4f}")
            
            return {
                **model_info,
                'evaluation_metrics': {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1},
                'test_loader': test_loader,
                'dataset': 'ImageNette',
                'num_classes': 10,
                'success': True
            }
            
        except Exception as e:
            error_msg = f"ImageNette validation failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'model_path': model_path,
                'error': error_msg,
                'success': False
            }
    
    def compare_models(self, model_paths: List[str], batch_size: int = 16) -> Dict:
        """
        Compare multiple trained ImageNette models.
        
        Args:
            model_paths: List of paths to saved models
            batch_size: Batch size for evaluation
            
        Returns:
            dict: Comparison results
        """
        print(f"\n{'='*70}")
        print(f"🔍 Comparing Multiple ImageNette Models")
        print(f"{'='*70}")
        
        results = []
        
        for model_path in model_paths:
            print(f"\n📊 Evaluating: {os.path.basename(model_path)}")
            result = self.validate_model(model_path, batch_size, verbose=False, show_visualization=False)
            results.append(result)
            
            if result['success']:
                metrics = result['evaluation_metrics']
                print(f"   ImageNette Accuracy: {metrics['accuracy']:.4f}")
            else:
                print(f"   Failed: {result['error']}")
        
        # Summary comparison
        print(f"\n📈 ImageNette Model Comparison Summary:")
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print(f"{'-'*70}")
        
        successful_results = [r for r in results if r['success']]
        successful_results.sort(key=lambda x: x['evaluation_metrics']['accuracy'], reverse=True)
        
        for result in successful_results:
            metrics = result['evaluation_metrics']
            model_name = result['model_name']
            print(f"{model_name:<20} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1']:<10.4f}")
        
        # Show failed models
        failed_results = [r for r in results if not r['success']]
        for result in failed_results:
            model_name = os.path.basename(result['model_path']).replace('.pt', '')
            print(f"{model_name:<20} {'FAILED':<10}")
        
        return {
            'results': results,
            'successful': successful_results,
            'failed': failed_results,
            'best_model': successful_results[0] if successful_results else None,
            'dataset': 'ImageNette',
            'num_classes': 10
        }
    
    def train_and_validate(self, model_name: str, config_name: str = ImageNetteTrainingConfigs.STANDARD) -> Dict:
        """
        Train and validate an ImageNette model in one command.
        
        Args:
            model_name: Model to train
            config_name: Configuration to use (must be from ImageNetteTrainingConfigs)
            
        Returns:
            dict: Combined training and validation results
            
        Raises:
            ValueError: If config_name is not valid
        """
        print(f"\n{'='*70}")
        print(f"⚡ Train & Validate ImageNette Model: {model_name} with {config_name}")
        print(f"{'='*70}")
        
        # Validate configuration
        if config_name not in self.AVAILABLE_CONFIGS:
            available_configs = ', '.join(self.AVAILABLE_CONFIGS)
            raise ValueError(f"Configuration '{config_name}' not available. Available configurations: {available_configs}")
        
        # Train model
        training_result = self.train_model(model_name, config_name)
        
        if not training_result['success']:
            return training_result
        
        # Validate model
        validation_result = self.validate_model(training_result['save_path'])
        
        return {
            'training': training_result,
            'validation': validation_result,
            'dataset': 'ImageNette',
            'num_classes': 10,
            'success': validation_result['success']
        }


if __name__ == "__main__":
    # Example usage
    trainer = ImageNetteModelTrainer()
    
    # Show available options
    trainer.list_available_models()
    trainer.list_available_configs()
    trainer.list_imagenette_classes()
    
    # Example: Train and validate a model using strongly typed config names
    # result = trainer.train_and_validate('resnet18', ImageNetteTrainingConfigs.TEST)
    
    # Example: Train multiple models
    # results = trainer.train_multiple_models(['resnet18', 'densenet121'], ImageNetteTrainingConfigs.STANDARD)
    
    # Example: Compare models
    # model_paths = ['./models/imagenette/resnet18_standard.pt', './models/imagenette/densenet121_standard.pt']
    # comparison = trainer.compare_models(model_paths)