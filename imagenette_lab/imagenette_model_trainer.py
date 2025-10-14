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

from domain.model.model_names import ModelNames

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.imagenet_models import ImageNetModels
from config.imagenette_classes import ImageNetteClasses
from data_eng.dataset_loader import load_imagenette, load_attacked_imagenette, load_attacked_imagenette_for_adversarial_training
from data_eng.io import load_model_imagenette, load_model_binary
from training.train import Training
from training.transfer.setup_pretraining import SetupPretraining
from imagenette_training_configs import ImageNetteTrainingConfigs
from evaluation.metrics import Metrics
from evaluation.validation import Validation
from evaluation.visualization import simple_visualize
from attacks.attack_names import AttackNames


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
        self.noise_detection_dir = './models/noise_detection'
        self.adversarial_models_dir = './models/imagenette_adversarial'
        self.AVAILABLE_MODELS = ImageNetteTrainingConfigs.AVAILABLE_MODELS
        self.AVAILABLE_CONFIGS = ImageNetteTrainingConfigs.ALL_CONFIGS
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.noise_detection_dir, exist_ok=True)
        os.makedirs(self.adversarial_models_dir, exist_ok=True)
        
        print(f"🚀 ImageNetteModelTrainer initialized on device: {self.device}")
        print(f"📁 Models will be saved to: {self.models_dir}")
        print(f"📁 Noise detection models will be saved to: {self.noise_detection_dir}")
        print(f"📁 Adversarial models will be saved to: {self.adversarial_models_dir}")
        print(f"🎯 Dataset: ImageNette (10 classes)")
    

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
    
    def train_noise_detection_model(
        self, 
        model_name: str,
        attacked_images_folder: str = "data/attacks/imagenette_models",
        clean_train_folder: str = "./data/imagenette/train",
        clean_test_folder: str = "./data/imagenette/val",
        batch_size: int = 32,
        learning_rate: float = 0.001,
        num_epochs: int = 20,
        early_stopping_patience: int = 7,
        scheduler_type: str = 'plateau',
        weight_decay: float = 0.0001,
        gradient_clip_norm: float = 1.0,
        verbose: bool = True
    ) -> Dict:
        """
        Train a binary classifier to detect adversarial noise in ImageNette images.
        
        This method trains a model to distinguish between clean and adversarial images
        using the attacked ImageNette dataset generated by attack_and_save_images_multiple.
        
        Args:
            model_name: Name of the model architecture to use
            attacked_images_folder: Folder containing attacked images from all models
            clean_train_folder: Folder containing clean ImageNette training images
            clean_test_folder: Folder containing clean ImageNette validation images
            batch_size: Batch size for training
            learning_rate: Initial learning rate
            num_epochs: Maximum number of training epochs
            early_stopping_patience: Epochs to wait before early stopping
            scheduler_type: Learning rate scheduler ('plateau', 'step', 'cosine')
            weight_decay: L2 regularization weight
            gradient_clip_norm: Gradient clipping threshold (None to disable)
            verbose: Whether to print detailed progress
            
        Returns:
            dict: Training results including model, metrics, and metadata
        """
        print(f"\n{'='*70}")
        print(f"🔍 Training Noise Detection Model: {model_name}")
        print(f"{'='*70}")
        
        try:
            # Validate model name
            if model_name not in self.AVAILABLE_MODELS:
                available_models = ', '.join(self.AVAILABLE_MODELS)
                raise ValueError(f"Model '{model_name}' not available. Available models: {available_models}")
            
            # Load attacked ImageNette dataset
            print(f"\n📁 Loading attacked ImageNette dataset for binary classification...")
            train_loader, test_loader = load_attacked_imagenette(
                attacked_images_folder=attacked_images_folder,
                clean_train_folder=clean_train_folder,
                clean_test_folder=clean_test_folder,
                batch_size=batch_size,
                shuffle=True
            )
            
            # Create fresh model instance
            print(f"\n🔧 Creating {model_name} model instance...")
            model = ImageNetModels.get_model(model_name)
            
            # Setup save path
            save_path = os.path.join(self.noise_detection_dir, f"{model_name}_noise_detector.pt")
            
            # Train noise detection model
            print(f"\n🚀 Starting noise detection training...")
            start_time = datetime.now()
            
            training_results = Training.train_imagenette_noise_detection(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                device=str(self.device),
                save_model_path=save_path,
                model_name=model_name,
                writer=None,  # TensorBoard writer will be created automatically
                setup_model=True,  # Automatically setup for binary classification
                validation_frequency=1,
                early_stopping_patience=early_stopping_patience,
                min_delta=0.001,
                scheduler_type=scheduler_type,
                scheduler_params=None,
                gradient_clip_norm=gradient_clip_norm,
                weight_decay=weight_decay,
                verbose=verbose
            )
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Print results summary
            print(f"\n✅ Noise detection training completed successfully!")
            print(f"   Training time: {training_time:.2f}s ({training_time/60:.2f} minutes)")
            print(f"   Best validation accuracy: {training_results['best_val_accuracy']:.2f}%")
            print(f"   Best precision: {training_results['best_precision']:.2f}%")
            print(f"   Best recall: {training_results['best_recall']:.2f}%")
            print(f"   Best F1 score: {training_results['best_f1']:.2f}%")
            print(f"   Model saved to: {save_path}")
            
            return {
                'model_name': model_name,
                'task': 'noise_detection',
                'training_results': training_results,
                'training_time': training_time,
                'save_path': save_path,
                'attacked_images_folder': attacked_images_folder,
                'clean_train_folder': clean_train_folder,
                'clean_test_folder': clean_test_folder,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'num_epochs': num_epochs,
                'best_val_accuracy': training_results['best_val_accuracy'],
                'best_precision': training_results['best_precision'],
                'best_recall': training_results['best_recall'],
                'best_f1': training_results['best_f1'],
                'success': True
            }
            
        except Exception as e:
            error_msg = f"Noise detection training failed: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                'model_name': model_name,
                'task': 'noise_detection',
                'error': error_msg,
                'success': False
            }
    
    def train_multiple_noise_detectors(
        self,
        model_names: List[str],
        attacked_images_folder: str = "data/attacks/imagenette_models",
        clean_train_folder: str = "./data/imagenette/train",
        clean_test_folder: str = "./data/imagenette/val",
        batch_size: int = 32,
        learning_rate: float = 0.001,
        num_epochs: int = 20
    ) -> List[Dict]:
        """
        Train multiple models for noise detection.
        
        Args:
            model_names: List of model names to train
            attacked_images_folder: Folder containing attacked images
            clean_train_folder: Folder containing clean training images
            clean_test_folder: Folder containing clean validation images
            batch_size: Batch size for training
            learning_rate: Learning rate for all models
            num_epochs: Number of epochs for all models
            
        Returns:
            List of training results for each model
        """
        print(f"\n{'='*70}")
        print(f"🚀 Training Multiple Noise Detection Models")
        print(f"{'='*70}")
        print(f"   Models: {len(model_names)}")
        print(f"   Attacked images folder: {attacked_images_folder}")
        
        results = []
        
        for i, model_name in enumerate(model_names, 1):
            print(f"\n📊 Model {i}/{len(model_names)}: {model_name}")
            
            result = self.train_noise_detection_model(
                model_name=model_name,
                attacked_images_folder=attacked_images_folder,
                clean_train_folder=clean_train_folder,
                clean_test_folder=clean_test_folder,
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                verbose=True
            )
            
            results.append(result)
            
            if result['success']:
                print(f"✅ {model_name}: {result['best_val_accuracy']:.2f}% accuracy, F1: {result['best_f1']:.2f}%")
            else:
                print(f"❌ {model_name}: Failed - {result['error']}")
        
        # Summary
        print(f"\n📈 Noise Detection Training Summary:")
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"   Successful: {len(successful)}/{len(results)}")
        print(f"   Failed: {len(failed)}/{len(results)}")
        
        if successful:
            print(f"\n🏆 Best Noise Detectors:")
            successful.sort(key=lambda x: x['best_f1'], reverse=True)
            for i, result in enumerate(successful[:5], 1):
                print(f"   {i}. {result['model_name']}: "
                      f"Acc={result['best_val_accuracy']:.2f}%, "
                      f"F1={result['best_f1']:.2f}%, "
                      f"Precision={result['best_precision']:.2f}%, "
                      f"Recall={result['best_recall']:.2f}%")
        
        return results
    
    def train_adversarial_model(
        self,
        model_name: str,
        attack_names: List[str] = None,
        learning_rate: float = 0.001,
        num_epochs: int = 20,
        batch_size: int = 32,
        adversarial_ratio: float = 0.5,
        attack_epsilon: float = 0.03,
        attack_alpha: float = 0.01,
        attack_steps: int = 10,
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
            model_name: Name of the model architecture to use
            attack_names: List of attacks for on-the-fly mode (required if use_preattacked_images=False)
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            adversarial_ratio: Ratio of adversarial to clean examples for on-the-fly mode (0.5 = 50/50)
            attack_epsilon: Maximum perturbation for on-the-fly attacks
            attack_alpha: Step size for iterative on-the-fly attacks
            attack_steps: Number of steps for iterative on-the-fly attacks
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
        print(f"\n{'='*70}")
        print(f"🛡️ Adversarial Training: {model_name}")
        print(f"{'='*70}")
        
        try:
            # Validate model name
            if model_name not in self.AVAILABLE_MODELS:
                available_models = ', '.join(self.AVAILABLE_MODELS)
                raise ValueError(f"Model '{model_name}' not available. Available models: {available_models}")
            
            # Validate attack names (only for on-the-fly mode)
            if not use_preattacked_images:
                if attack_names is None or len(attack_names) == 0:
                    raise ValueError("attack_names must be provided for on-the-fly adversarial training")
                
                available_attacks = AttackNames().all_attack_names
                for attack_name in attack_names:
                    if attack_name not in available_attacks:
                        raise ValueError(f"Attack '{attack_name}' not available. Available attacks: {', '.join(available_attacks)}")
            
            print(f"📊 Adversarial Training Configuration:")
            print(f"   Model: {model_name}")
            print(f"   Mode: {'Pre-attacked Images' if use_preattacked_images else 'On-the-fly Generation'}")
            if not use_preattacked_images:
                print(f"   Attacks: {', '.join(attack_names)}")
                print(f"   Adversarial ratio: {adversarial_ratio:.1%}")
                print(f"   Attack epsilon: {attack_epsilon}")
            print(f"   Learning Rate: {learning_rate}")
            print(f"   Epochs: {num_epochs}")
            print(f"   Batch size: {batch_size}")
            
            # Create fresh model instance
            print(f"\n🔧 Creating {model_name} model instance...")
            model = ImageNetModels.get_model(model_name)
            
            # Load dataset based on mode
            print(f"\n📁 Loading dataset...")
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
            print(f"\n🔥 Starting adversarial training...")
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
                model_name=model_name,
                writer=None,  # Will be created automatically
                setup_model=True,  # Automatically setup for ImageNette
                validation_frequency=1,
                early_stopping_patience=early_stopping_patience,
                min_delta=0.001,
                scheduler_type=scheduler_type,
                scheduler_params=None,
                gradient_clip_norm=gradient_clip_norm,
                weight_decay=weight_decay,
                attack_epsilon=attack_epsilon,
                attack_alpha=attack_alpha,
                attack_steps=attack_steps,
                adversarial_ratio=adversarial_ratio,
                use_preattacked_images=use_preattacked_images,
                verbose=verbose
            )
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Print results summary
            print(f"\n✅ Adversarial training completed successfully!")
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
        model_names: List[str],
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
            model_names: List of model names to train
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
        print(f"🚀 Training Multiple Adversarial Models")
        print(f"{'='*70}")
        print(f"   Models: {len(model_names)}")
        print(f"   Mode: {'Pre-attacked Images' if use_preattacked_images else 'On-the-fly Generation'}")
        if not use_preattacked_images:
            print(f"   Attacks: {', '.join(attack_names)}")
            print(f"   Adversarial Ratio: {adversarial_ratio:.1%}")
        print(f"   Epochs: {num_epochs}")
        
        results = []
        
        for i, model_name in enumerate(model_names, 1):
            print(f"\n📊 Model {i}/{len(model_names)}: {model_name}")
            
            result = self.train_adversarial_model(
                model_name=model_name,
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
        print(f"\n📈 Adversarial Training Summary:")
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"   Successful: {len(successful)}/{len(results)}")
        print(f"   Failed: {len(failed)}/{len(results)}")
        
        if successful:
            print(f"\n🏆 Best Adversarially Trained Models (by validation accuracy):")
            successful.sort(key=lambda x: x['best_val_accuracy'], reverse=True)
            for i, result in enumerate(successful[:5], 1):
                print(f"   {i}. {result['model_name']}: "
                      f"Val Acc={result['best_val_accuracy']:.2f}%")
        
        return results

if __name__ == "__main__":
    # Example usage
    trainer = ImageNetteModelTrainer()
    
    # Show available options
    trainer.list_available_models()
    trainer.list_available_configs()
    trainer.list_imagenette_classes()
    
    # ===== ImageNette Classification Examples =====
    # Example: Train and validate a model using strongly typed config names
    # result = trainer.train_and_validate('resnet18', ImageNetteTrainingConfigs.TEST)
    
    # Example: Train multiple models
    # results = trainer.train_multiple_models(['resnet18', 'densenet121'], ImageNetteTrainingConfigs.STANDARD)
    
    # Example: Compare models
    # model_paths = ['./models/imagenette/resnet18_standard.pt', './models/imagenette/densenet121_standard.pt']
    # comparison = trainer.compare_models(model_paths)
    
    # ===== Noise Detection Examples =====
    # Example: Train single noise detection model
    # result = trainer.train_noise_detection_model(
    #     model_name='resnet18',
    #     attacked_images_folder='data/attacks/imagenette_models',
    #     clean_train_folder='./data/imagenette/train',
    #     clean_test_folder='./data/imagenette/val',
    #     batch_size=32,
    #     learning_rate=0.001,
    #     num_epochs=20
    # )
    
    # Example: Train multiple noise detection models
    # models_names = [
    #     ModelNames().resnet18,
    #     ModelNames().vgg16,
    #     ModelNames().densenet121,
    #     ModelNames().mobilenet_v2,
    #     ModelNames().efficientnet_b0
    # ]
    
    # results = trainer.train_multiple_noise_detectors(
    #     model_names=models_names,
    #     attacked_images_folder='data/attacks/imagenette_models',
    #     batch_size=32,
    #     num_epochs=20
    # )
    
    # ===== Adversarial Training Examples =====
    # Example 1: On-the-fly adversarial training (generates attacks during training)
    # result = trainer.train_adversarial_model(
    #     model_name='resnet18',
    #     attack_names=['FGSM', 'PGD', 'BIM'],
    #     learning_rate=0.001,
    #     num_epochs=20,
    #     batch_size=32,
    #     adversarial_ratio=0.5,  # 50% adversarial, 50% clean
    #     use_preattacked_images=False  # Generate on-the-fly
    # )
    
    # Example 2: Train on pre-attacked images (faster, uses saved images)
    # result = trainer.train_adversarial_model(
    #     model_name='resnet18',
    #     learning_rate=0.001,
    #     num_epochs=20,
    #     batch_size=128,
    #     use_preattacked_images=True,  # Use pre-saved attacked images
    #     attacked_images_folder='data/attacks/imagenette_models'
    # )
    
    # Example 3: Train multiple models with adversarial training
    # model_names = [
    #     ModelNames().resnet18,
    #     ModelNames().densenet121,
    #     ModelNames().mobilenet_v2
    # ]
    # 
    # # On-the-fly mode
    # results = trainer.train_multiple_adversarial_models(
    #     model_names=model_names,
    #     attack_names=['FGSM', 'PGD', 'BIM'],
    #     learning_rate=0.001,
    #     num_epochs=20,
    #     batch_size=32,
    #     adversarial_ratio=0.5,
    #     use_preattacked_images=False
    # )
    # 
    # # Pre-attacked images mode
    model_names = [
        ModelNames().resnet18,
        ModelNames().vgg16,
        ModelNames().densenet121,
        ModelNames().mobilenet_v2,
        ModelNames().efficientnet_b0
    ]
    results = trainer.train_multiple_adversarial_models(
        model_names=model_names,
        learning_rate=0.001,
        num_epochs=20,
        batch_size=64,
        use_preattacked_images=False,
        attacked_images_folder='data/attacks/imagenette_models'
    )