#!/usr/bin/env python3
"""
ImageNette Validator - Comprehensive validation system for trained ImageNette models.

This module validates all trained ImageNette models and saves detailed metrics
in CSV format for analysis and comparison.

Usage:
    from imagenette_lab.imagenette_validator import ImageNetteValidator
    
    validator = ImageNetteValidator()
    validator.validate_all_models()
    validator.generate_summary_report()
"""

import torch
import os
import sys
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np

from imagenette_lab.imagenette_training_configs import ImageNetteTrainingConfigs

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.imagenette_classes import ImageNetteClasses
from domain.model.model_names import ModelNames
from data_eng.dataset_loader import load_imagenette
from data_eng.io import load_model_imagenette
from evaluation.metrics import Metrics

class ImageNetteValidator:
    """
    Comprehensive validator for trained ImageNette models.
    
    This class validates all trained models and saves detailed metrics
    including overall performance and per-class performance.
    """
    
    def __init__(self, models_dir: str = './models/imagenette', 
                 results_dir: str = './results/imagenette_trained_models',
                 device: str = 'auto'):
        """
        Initialize the ImageNetteValidator.
        
        Args:
            models_dir: Directory containing trained models
            results_dir: Directory to save validation results
            device: Device to use for validation
        """
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.models_dir = models_dir
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Available models to validate
        self.available_models = ImageNetteTrainingConfigs.AVAILABLE_MODELS
        
        # ImageNette classes
        self.imagenette_classes = ImageNetteClasses.get_classes()
        
        print(f"🚀 ImageNetteValidator initialized on device: {self.device}")
        print(f"📁 Models directory: {self.models_dir}")
        print(f"📊 Results directory: {self.results_dir}")
        print(f"🎯 Dataset: ImageNette (10 classes)")
        print(f"🤖 Models to validate: {len(self.available_models)}")
    
    def _get_model_path(self, model_name: str) -> str:
        """
        Get the path to a trained model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            str: Path to the model file
        """
        return os.path.join(self.models_dir, f"{model_name}_advanced.pt")
    
    def _model_exists(self, model_name: str) -> bool:
        """
        Check if a trained model exists.
        
        Args:
            model_name: Name of the model
            
        Returns:
            bool: True if model exists, False otherwise
        """
        model_path = self._get_model_path(model_name)
        return os.path.exists(model_path)
    
    def _get_available_trained_models(self) -> List[str]:
        """
        Get list of models that have been trained.
        
        Returns:
            List[str]: List of available trained model names
        """
        available_models = []
        for model_name in self.available_models:
            if self._model_exists(model_name):
                available_models.append(model_name)
            else:
                print(f"⚠️ Model not found: {model_name}_advanced.pt")
        
        return available_models
    
    def _calculate_per_class_metrics(self, model, test_loader) -> Dict:
        """
        Calculate per-class metrics for a model.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            
        Returns:
            dict: Per-class metrics
        """
        model.eval()
        device = next(model.parameters()).device
        
        # Initialize counters
        class_correct = [0] * 10
        class_total = [0] * 10
        class_predictions = [[] for _ in range(10)]
        class_targets = [[] for _ in range(10)]
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                
                # Update counters
                for i in range(len(target)):
                    label = target[i].item()
                    prediction = pred[i].item()
                    
                    class_total[label] += 1
                    class_predictions[label].append(prediction)
                    class_targets[label].append(label)
                    
                    if prediction == label:
                        class_correct[label] += 1
        
        # Calculate per-class metrics
        per_class_metrics = {}
        for i in range(10):
            if class_total[i] > 0:
                accuracy = class_correct[i] / class_total[i]
                
                # Calculate precision, recall, F1 for this class
                true_positives = class_correct[i]
                false_positives = sum(1 for p in class_predictions[i] if p == i) - true_positives
                false_negatives = class_total[i] - true_positives
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                per_class_metrics[self.imagenette_classes[i]] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'samples': class_total[i]
                }
            else:
                per_class_metrics[self.imagenette_classes[i]] = {
                    'accuracy': 0,
                    'precision': 0,
                    'recall': 0,
                    'f1': 0,
                    'samples': 0
                }
        
        return per_class_metrics
    
    def validate_model(self, model_name: str) -> Dict:
        """
        Validate a single trained model.
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            dict: Validation results
        """
        print(f"\n🔍 Validating {model_name}...")
        
        try:
            # Check if model exists
            if not self._model_exists(model_name):
                raise FileNotFoundError(f"Model {model_name}_advanced.pt not found")
            
            # Load model
            model_path = self._get_model_path(model_name)
            model_info = load_model_imagenette(model_path, device=str(self.device), verbose=False)
            
            if not model_info['success']:
                raise RuntimeError(f"Failed to load model: {model_info['error']}")
            
            model = model_info['model']
            
            # Create test dataset (full dataset)
            _, test_loader = load_imagenette(batch_size=32, test_subset_size=-1)
            
            # Calculate overall metrics
            print(f"   Calculating overall metrics...")
            acc, prec, rec, f1 = Metrics.evaluate_model_torchmetrics(
                model, test_loader, 10, verbose=False
            )
            
            # Calculate per-class metrics
            print(f"   Calculating per-class metrics...")
            per_class_metrics = self._calculate_per_class_metrics(model, test_loader)
            
            # Get checkpoint info
            checkpoint = model_info['checkpoint']
            
            result = {
                'model_name': model_name,
                'overall_accuracy': acc,
                'overall_precision': prec,
                'overall_recall': rec,
                'overall_f1': f1,
                'per_class_metrics': per_class_metrics,
                'training_epoch': checkpoint.get('epoch', 'Unknown'),
                'training_val_accuracy': checkpoint.get('val_accuracy', 'Unknown'),
                'validation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'success': True
            }
            
            print(f"   ✅ {model_name}: Overall Accuracy = {acc:.4f}")
            return result
            
        except Exception as e:
            error_msg = f"Validation failed for {model_name}: {str(e)}"
            print(f"   ❌ {error_msg}")
            return {
                'model_name': model_name,
                'error': error_msg,
                'success': False
            }
    
    def validate_all_models(self) -> List[Dict]:
        """
        Validate all available trained models.
        
        Returns:
            List[Dict]: List of validation results for all models
        """
        print(f"\n{'='*70}")
        print(f"🚀 Validating All Trained ImageNette Models")
        print(f"{'='*70}")
        
        # Get available trained models
        available_trained_models = self._get_available_trained_models()
        
        if not available_trained_models:
            print("❌ No trained models found!")
            return []
        
        print(f"📊 Found {len(available_trained_models)} trained models:")
        for model_name in available_trained_models:
            print(f"   - {model_name}")
        
        # Validate each model
        results = []
        for model_name in available_trained_models:
            result = self.validate_model(model_name)
            results.append(result)
        
        # Summary
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"\n📈 Validation Summary:")
        print(f"   Successful: {len(successful)}/{len(results)}")
        print(f"   Failed: {len(failed)}/{len(results)}")
        
        if successful:
            print(f"\n🏆 Model Performance Ranking:")
            successful.sort(key=lambda x: x['overall_accuracy'], reverse=True)
            for i, result in enumerate(successful, 1):
                print(f"   {i}. {result['model_name']}: {result['overall_accuracy']:.4f}")
        
        return results
    
    def save_all_models_summary(self, results: List[Dict]) -> str:
        """
        Save summary results for all models in a single CSV file.
        
        Args:
            results: List of validation results
            
        Returns:
            str: Path to saved file
        """
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            print("❌ No successful results to save")
            return ""
        
        # Prepare data for summary CSV
        summary_data = []
        for result in successful_results:
            summary_data.append({
                'model_name': result['model_name'],
                'overall_accuracy': result['overall_accuracy'],
                'overall_precision': result['overall_precision'],
                'overall_recall': result['overall_recall'],
                'overall_f1': result['overall_f1'],
                'training_epoch': result['training_epoch'],
                'training_val_accuracy': result['training_val_accuracy'],
                'validation_timestamp': result['validation_timestamp']
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(summary_data)
        df = df.sort_values('overall_accuracy', ascending=False)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"imagenette_models_summary_{timestamp}.csv"
        filepath = os.path.join(self.results_dir, filename)
        
        df.to_csv(filepath, index=False)
        print(f"📊 Summary results for all models saved to: {filepath}")
        
        return filepath
    
    def save_model_per_class_results(self, result: Dict) -> str:
        """
        Save per-class results for a single model to CSV.
        
        Args:
            result: Validation result for a single model
            
        Returns:
            str: Path to saved file
        """
        if not result['success']:
            print(f"❌ Cannot save results for failed model: {result['model_name']}")
            return ""
        
        # Prepare data for per-class CSV
        per_class_data = []
        model_name = result['model_name']
        per_class_metrics = result['per_class_metrics']
        
        for class_name, metrics in per_class_metrics.items():
            per_class_data.append({
                'model_name': model_name,
                'class_name': class_name,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'samples': metrics['samples']
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(per_class_data)
        df = df.sort_values('accuracy', ascending=False)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{result['model_name']}_per_class_{timestamp}.csv"
        filepath = os.path.join(self.results_dir, filename)
        
        df.to_csv(filepath, index=False)
        print(f"📊 Per-class results for {result['model_name']} saved to: {filepath}")
        
        return filepath
    
    
    def run_full_validation(self) -> Dict[str, str]:
        """
        Run complete validation pipeline.
        
        Returns:
            dict: Paths to generated files
        """
        print(f"\n{'='*70}")
        print(f"🚀 Running Full ImageNette Validation Pipeline")
        print(f"{'='*70}")
        
        # Validate all models
        results = self.validate_all_models()
        
        if not results:
            print("❌ No models to validate")
            return {}
        
        # Save single summary file for all models
        summary_file = self.save_all_models_summary(results)
        
        # Save per-class results for each model
        per_class_files = []
        for result in results:
            if result['success']:
                per_class_file = self.save_model_per_class_results(result)
                if per_class_file:
                    per_class_files.append(per_class_file)
        
        print(f"\n📁 Generated 1 summary file and {len(per_class_files)} per-class files")
        
        return {
            'summary_csv': summary_file,
            'per_class_csv_files': per_class_files
        }


if __name__ == "__main__":
    # Example usage
    validator = ImageNetteValidator()
    
    # Run full validation pipeline
    files = validator.run_full_validation()
    
    print(f"\n✅ Validation complete! Generated files:")
    
    # Display summary file
    if files.get('summary_csv'):
        print(f"\n📊 Summary file:")
        print(f"   📄 {files['summary_csv']}")
    
    # Display per-class files
    if files.get('per_class_csv_files'):
        print(f"\n📊 Per-class files ({len(files['per_class_csv_files'])}):")
        for file_path in files['per_class_csv_files']:
            print(f"   📄 {file_path}")
    
    # Example: Validate specific model
    # result = validator.validate_model('resnet18')
    # print(f"ResNet18 accuracy: {result['overall_accuracy']:.4f}")
    
    # Example: Get available trained models
    # available_models = validator._get_available_trained_models()
    # print(f"Available models: {available_models}")
