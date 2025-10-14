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
from data_eng.io import load_model_imagenette, load_model_binary
from data_eng.dataset_loader import load_attacked_imagenette
from evaluation.metrics import Metrics

class ImageNetteValidator:
    """
    Comprehensive validator for trained ImageNette models.
    
    This class validates all trained models and saves detailed metrics
    including overall performance and per-class performance.
    """
    
    def __init__(self, models_dir: str = './models/imagenette', 
                 results_dir: str = './results/imagenette_trained_models',
                 noise_detection_models_dir: str = './models/noise_detection',
                 noise_detection_results_dir: str = './results/noise_detection_models',
                 adversarial_models_dir: str = './models/imagenette_adversarial',
                 adversarial_results_dir: str = './results/adversarial_models',
                 device: str = 'auto'):
        """
        Initialize the ImageNetteValidator.
        
        Args:
            models_dir: Directory containing trained ImageNette classification models
            results_dir: Directory to save ImageNette validation results
            noise_detection_models_dir: Directory containing trained noise detection models
            noise_detection_results_dir: Directory to save noise detection validation results
            device: Device to use for validation
        """
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.noise_detection_models_dir = noise_detection_models_dir
        self.noise_detection_results_dir = noise_detection_results_dir
        self.adversarial_models_dir = adversarial_models_dir
        self.adversarial_results_dir = adversarial_results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.noise_detection_results_dir, exist_ok=True)
        os.makedirs(self.adversarial_results_dir, exist_ok=True)
        
        # Available models to validate
        self.available_models = ImageNetteTrainingConfigs.AVAILABLE_MODELS
        
        # ImageNette classes
        self.imagenette_classes = ImageNetteClasses.get_classes()
        
        print(f"🚀 ImageNetteValidator initialized on device: {self.device}")
        print(f"📁 ImageNette models directory: {self.models_dir}")
        print(f"📁 Noise detection models directory: {self.noise_detection_models_dir}")
        print(f"📊 ImageNette results directory: {self.results_dir}")
        print(f"📊 Noise detection results directory: {self.noise_detection_results_dir}")
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
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'per_class_metrics': per_class_metrics,
                'training_epoch': checkpoint.get('epoch', 'Unknown'),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
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
            successful.sort(key=lambda x: x['accuracy'], reverse=True)
            for i, result in enumerate(successful, 1):
                print(f"   {i}. {result['model_name']}: {result['accuracy']:.4f}")
        
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
                'accuracy': result['accuracy'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1': result['f1'],
                'training_epoch': result['training_epoch'],
                'timestamp': result['timestamp']
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(summary_data)
        df = df.sort_values('accuracy', ascending=False)
        
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
    
    # ===== NOISE DETECTION VALIDATION METHODS =====
    
    def _get_noise_detection_model_path(self, model_name: str) -> str:
        """
        Get the path to a trained noise detection model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            str: Path to the noise detection model file
        """
        return os.path.join(self.noise_detection_models_dir, f"{model_name}_noise_detector.pt")
    
    def _noise_detection_model_exists(self, model_name: str) -> bool:
        """
        Check if a trained noise detection model exists.
        
        Args:
            model_name: Name of the model
            
        Returns:
            bool: True if model exists, False otherwise
        """
        model_path = self._get_noise_detection_model_path(model_name)
        return os.path.exists(model_path)
    
    def _get_available_noise_detection_models(self) -> List[str]:
        """
        Get list of noise detection models that have been trained.
        
        Returns:
            List[str]: List of available trained noise detection model names
        """
        available_models = []
        for model_name in self.available_models:
            if self._noise_detection_model_exists(model_name):
                available_models.append(model_name)
            else:
                print(f"⚠️ Noise detection model not found: {model_name}_noise_detector.pt")
        
        return available_models
    
    def validate_noise_detection_model(self, model_name: str, 
                                     attacked_images_folder: str = "data/attacks/imagenette_models",
                                     clean_test_folder: str = "./data/imagenette/val",
                                     batch_size: int = 32) -> Dict:
        """
        Validate a single trained noise detection model.
        
        Args:
            model_name: Name of the noise detection model to validate
            attacked_images_folder: Folder containing attacked images
            clean_test_folder: Folder containing clean test images
            batch_size: Batch size for validation
            
        Returns:
            dict: Validation results including confusion matrix and metrics
        """
        print(f"\n🔍 Validating noise detection model: {model_name}...")
        
        try:
            # Check if model exists
            if not self._noise_detection_model_exists(model_name):
                raise FileNotFoundError(f"Noise detection model {model_name}_noise_detector.pt not found")
            
            # Load model
            model_path = self._get_noise_detection_model_path(model_name)
            model_info = load_model_binary(model_path, device=str(self.device), verbose=False)
            
            if not model_info['success']:
                raise RuntimeError(f"Failed to load noise detection model: {model_info['error']}")
            
            model = model_info['model']
            
            # Load test dataset
            print(f"   Loading test dataset...")
            _, test_loader = load_attacked_imagenette(
                attacked_images_folder=attacked_images_folder,
                clean_train_folder="./data/imagenette/train",  # Not used for test
                clean_test_folder=clean_test_folder,
                batch_size=batch_size,
                shuffle=False
            )
            
            # Evaluate model
            print(f"   Running evaluation...")
            model.eval()
            correct = 0
            total = 0
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device).float().unsqueeze(1)
                    
                    outputs = model(images)
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Confusion matrix
                    true_positives += ((predicted == 1) & (labels == 1)).sum().item()
                    false_positives += ((predicted == 1) & (labels == 0)).sum().item()
                    true_negatives += ((predicted == 0) & (labels == 0)).sum().item()
                    false_negatives += ((predicted == 0) & (labels == 1)).sum().item()
            
            # Calculate metrics
            accuracy = 100. * correct / total
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Get checkpoint info
            checkpoint = model_info['checkpoint']
            
            result = {
                'model_name': model_name,
                'task': 'noise_detection',
                'accuracy': accuracy,
                'precision': precision * 100,
                'recall': recall * 100,
                'f1': f1 * 100,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'true_negatives': true_negatives,
                'false_negatives': false_negatives,
                'total_samples': total,
                'training_epoch': checkpoint.get('epoch', 'Unknown'),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'success': True
            }
            
            print(f"   ✅ {model_name}: Accuracy = {accuracy:.2f}%, F1 = {f1*100:.2f}%")
            return result
            
        except Exception as e:
            error_msg = f"Noise detection validation failed for {model_name}: {str(e)}"
            print(f"   ❌ {error_msg}")
            return {
                'model_name': model_name,
                'task': 'noise_detection',
                'error': error_msg,
                'success': False
            }
    
    def validate_all_noise_detection_models(self) -> List[Dict]:
        """
        Validate all available trained noise detection models.
        
        Returns:
            List[Dict]: List of validation results for all noise detection models
        """
        print(f"\n{'='*70}")
        print(f"🚀 Validating All Trained Noise Detection Models")
        print(f"{'='*70}")
        
        # Get available trained noise detection models
        available_models = self._get_available_noise_detection_models()
        
        if not available_models:
            print("❌ No trained noise detection models found!")
            return []
        
        print(f"📊 Found {len(available_models)} trained noise detection models:")
        for model_name in available_models:
            print(f"   - {model_name}")
        
        # Validate each model
        results = []
        for model_name in available_models:
            result = self.validate_noise_detection_model(model_name)
            results.append(result)
        
        # Summary
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"\n📈 Noise Detection Validation Summary:")
        print(f"   Successful: {len(successful)}/{len(results)}")
        print(f"   Failed: {len(failed)}/{len(results)}")
        
        if successful:
            print(f"\n🏆 Noise Detection Model Performance Ranking:")
            successful.sort(key=lambda x: x['f1'], reverse=True)
            for i, result in enumerate(successful, 1):
                print(f"   {i}. {result['model_name']}: "
                      f"Acc={result['accuracy']:.2f}%, "
                      f"F1={result['f1']:.2f}%, "
                      f"Precision={result['precision']:.2f}%, "
                      f"Recall={result['recall']:.2f}%")
        
        return results
    
    def save_noise_detection_summary(self, results: List[Dict]) -> str:
        """
        Save summary results for all noise detection models in a single CSV file.
        
        Args:
            results: List of noise detection validation results
            
        Returns:
            str: Path to saved file
        """
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            print("❌ No successful noise detection results to save")
            return ""
        
        # Prepare data for summary CSV
        summary_data = []
        for result in successful_results:
            summary_data.append({
                'model_name': result['model_name'],
                'task': result['task'],
                'accuracy': result['accuracy'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1': result['f1'],
                'true_positives': result['true_positives'],
                'false_positives': result['false_positives'],
                'true_negatives': result['true_negatives'],
                'false_negatives': result['false_negatives'],
                'total_samples': result['total_samples'],
                'training_epoch': result['training_epoch'],
                'timestamp': result['timestamp']
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(summary_data)
        df = df.sort_values('f1', ascending=False)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"noise_detection_models_summary_{timestamp}.csv"
        filepath = os.path.join(self.noise_detection_results_dir, filename)
        
        df.to_csv(filepath, index=False)
        print(f"📊 Noise detection summary results for all models saved to: {filepath}")
        
        return filepath
    
    def run_noise_detection_validation(self) -> Dict[str, str]:
        """
        Run complete noise detection validation pipeline.
        
        Returns:
            dict: Path to generated summary file
        """
        print(f"\n{'='*70}")
        print(f"🚀 Running Full Noise Detection Validation Pipeline")
        print(f"{'='*70}")
        
        # Validate all noise detection models
        results = self.validate_all_noise_detection_models()
        
        if not results:
            print("❌ No noise detection models to validate")
            return {}
        
        # Save summary file for all models
        summary_file = self.save_noise_detection_summary(results)
        
        print(f"\n📁 Generated noise detection summary file")
        
        return {
            'noise_detection_summary_csv': summary_file
        }
    
    def validate_adversarial_model(
        self,
        model_name: str,
        attack_names: List[str],
        batch_size: int = 32,
        attack_epsilon: float = 0.03,
        attack_alpha: float = 0.01,
        attack_steps: int = 10
    ) -> Dict:
        """
        Validate a trained adversarial model on clean and adversarial test sets.
        
        Args:
            model_name: Name of the adversarial model to validate
            attack_names: List of attack names to test robustness
            batch_size: Batch size for validation
            attack_epsilon: Maximum perturbation for attacks
            attack_alpha: Step size for iterative attacks
            attack_steps: Number of steps for iterative attacks
            
        Returns:
            dict: Validation results including clean and adversarial accuracy
        """
        from attacks.attack_factory import AttackFactory
        
        print(f"\n🛡️ Validating adversarial model: {model_name}...")
        
        try:
            # Load model (try both pre-attacked and on-the-fly versions)
            model_path_preattacked = os.path.join(self.adversarial_models_dir, f"{model_name}_adv_preattacked.pt")
            model_path_onthefly = os.path.join(self.adversarial_models_dir, f"{model_name}_adv_onthefly.pt")
            
            if os.path.exists(model_path_preattacked):
                model_path = model_path_preattacked
                training_mode = "preattacked"
            elif os.path.exists(model_path_onthefly):
                model_path = model_path_onthefly
                training_mode = "onthefly"
            else:
                raise FileNotFoundError(f"Adversarial model not found. Tried:\n  {model_path_preattacked}\n  {model_path_onthefly}")
            
            print(f"   Loading model from: {model_path}")
            model_info = load_model_imagenette(model_path, device=str(self.device), verbose=False)
            
            if not model_info['success']:
                raise RuntimeError(f"Failed to load adversarial model: {model_info['error']}")
            
            model = model_info['model']
            model.eval()
            
            # Load clean test dataset
            print(f"   Loading clean test dataset...")
            _, test_loader = load_imagenette(batch_size=batch_size, shuffle=False)
            
            # Evaluate on clean images
            print(f"   Evaluating on clean images...")
            clean_correct = 0
            clean_total = 0
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    clean_total += labels.size(0)
                    clean_correct += (predicted == labels).sum().item()
            
            clean_accuracy = 100. * clean_correct / clean_total
            print(f"   ✅ Clean accuracy: {clean_accuracy:.2f}%")
            
            # Evaluate on adversarial images for each attack
            attack_accuracies = {}
            
            for attack_name in attack_names:
                print(f"   Evaluating robustness against {attack_name}...")
                
                adv_correct = 0
                adv_total = 0
                
                for images, labels in test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    # Generate adversarial examples
                    attack = AttackFactory.create_attack(
                        attack_name=attack_name,
                        model=model,
                        eps=attack_epsilon,
                        alpha=attack_alpha,
                        steps=attack_steps
                    )
                    
                    with torch.enable_grad():
                        adv_images = attack(images, labels)
                    
                    # Evaluate
                    with torch.no_grad():
                        outputs = model(adv_images)
                        _, predicted = torch.max(outputs, 1)
                        adv_total += labels.size(0)
                        adv_correct += (predicted == labels).sum().item()
                
                attack_accuracy = 100. * adv_correct / adv_total
                attack_accuracies[attack_name] = attack_accuracy
                print(f"   ✅ {attack_name} robustness: {attack_accuracy:.2f}%")
            
            # Calculate average adversarial accuracy
            avg_adv_accuracy = sum(attack_accuracies.values()) / len(attack_accuracies) if attack_accuracies else 0.0
            
            result = {
                'model_name': model_name,
                'task': 'adversarial_validation',
                'training_mode': training_mode,
                'clean_accuracy': clean_accuracy,
                'adversarial_accuracy': avg_adv_accuracy,
                'attack_accuracies': attack_accuracies,
                'total_samples': clean_total,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'success': True
            }
            
            print(f"   📊 Summary: Mode={training_mode}, Clean={clean_accuracy:.2f}%, Avg Adv={avg_adv_accuracy:.2f}%")
            return result
            
        except Exception as e:
            error_msg = f"Adversarial validation failed for {model_name}: {str(e)}"
            print(f"   ❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                'model_name': model_name,
                'task': 'adversarial_validation',
                'error': error_msg,
                'success': False
            }
    
    def validate_all_adversarial_models(
        self,
        attack_names: List[str],
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Validate all trained adversarial models.
        
        Args:
            attack_names: List of attacks to test robustness
            batch_size: Batch size for validation
            
        Returns:
            List of validation results for each model
        """
        print(f"\n{'='*70}")
        print(f"🛡️ Validating All Adversarial Models")
        print(f"{'='*70}")
        
        # Get all adversarial models
        if not os.path.exists(self.adversarial_models_dir):
            print(f"⚠️ Adversarial models directory not found: {self.adversarial_models_dir}")
            return []
        
        model_files = [f for f in os.listdir(self.adversarial_models_dir) if f.endswith('.pt')]
        
        if not model_files:
            print(f"⚠️ No adversarial models found in {self.adversarial_models_dir}")
            return []
        
        print(f"   Found {len(model_files)} adversarial models")
        
        # Extract unique model names (handle both _preattacked and _onthefly suffixes)
        model_names_set = set()
        for model_file in model_files:
            # Remove _adv_preattacked.pt or _adv_onthefly.pt
            model_name = model_file.replace('_adv_preattacked.pt', '').replace('_adv_onthefly.pt', '')
            model_names_set.add(model_name)
        
        results = []
        
        for i, model_name in enumerate(sorted(model_names_set), 1):
            print(f"\n📊 Validating {i}/{len(model_names_set)}: {model_name}")
            
            result = self.validate_adversarial_model(
                model_name=model_name,
                attack_names=attack_names,
                batch_size=batch_size
            )
            
            results.append(result)
        
        # Summary
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"\n📈 Adversarial Validation Summary:")
        print(f"   Successful: {len(successful)}/{len(results)}")
        print(f"   Failed: {len(failed)}/{len(results)}")
        
        if successful:
            print(f"\n🏆 Best Adversarial Models (by average adversarial accuracy):")
            successful.sort(key=lambda x: x['adversarial_accuracy'], reverse=True)
            for i, result in enumerate(successful[:5], 1):
                print(f"   {i}. {result['model_name']}: "
                      f"Clean={result['clean_accuracy']:.2f}%, "
                      f"Avg Adv={result['adversarial_accuracy']:.2f}%")
        
        return results
    
    def save_adversarial_validation_summary(self, results: List[Dict]) -> str:
        """
        Save adversarial validation results to CSV.
        
        Args:
            results: List of validation results from validate_all_adversarial_models
            
        Returns:
            str: Path to saved CSV file
        """
        import csv
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"adversarial_models_summary_{timestamp}.csv"
        filepath = os.path.join(self.adversarial_results_dir, filename)
        
        with open(filepath, 'w', newline='') as f:
            # Get all attack names from first successful result
            attack_names = []
            for result in results:
                if result['success'] and 'attack_accuracies' in result:
                    attack_names = list(result['attack_accuracies'].keys())
                    break
            
            # Build fieldnames
            fieldnames = ['model_name', 'training_mode', 'clean_accuracy', 'avg_adversarial_accuracy']
            for attack_name in attack_names:
                fieldnames.append(f'{attack_name}_accuracy')
            fieldnames.extend(['total_samples', 'timestamp'])
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                if result['success']:
                    row = {
                        'model_name': result['model_name'],
                        'training_mode': result.get('training_mode', 'unknown'),
                        'clean_accuracy': result['clean_accuracy'],
                        'avg_adversarial_accuracy': result['adversarial_accuracy'],
                        'total_samples': result['total_samples'],
                        'timestamp': result['timestamp']
                    }
                    
                    # Add individual attack accuracies
                    for attack_name in attack_names:
                        row[f'{attack_name}_accuracy'] = result['attack_accuracies'].get(attack_name, 0.0)
                    
                    writer.writerow(row)
        
        print(f"💾 Saved adversarial validation summary to: {filepath}")
        return filepath
    
    def run_adversarial_validation(
        self,
        attack_names: List[str],
        batch_size: int = 32
    ) -> Dict:
        """
        Run complete validation pipeline for adversarial models.
        
        Args:
            attack_names: List of attacks to test robustness
            batch_size: Batch size for validation
            
        Returns:
            dict: Paths to generated files
        """
        print(f"\n{'='*70}")
        print(f"🛡️ Running Adversarial Models Validation Pipeline")
        print(f"{'='*70}")
        
        # Validate all adversarial models
        results = self.validate_all_adversarial_models(
            attack_names=attack_names,
            batch_size=batch_size
        )
        
        # Save summary
        summary_file = self.save_adversarial_validation_summary(results)
        
        print(f"\n📁 Generated adversarial validation summary file")
        
        return {
            'adversarial_summary_csv': summary_file
        }
    
    def run_complete_validation(self) -> Dict[str, str]:
        """
        Run complete validation pipeline for both ImageNette classification and noise detection models.
        
        Returns:
            dict: Paths to all generated files
        """
        print(f"\n{'='*70}")
        print(f"🚀 Running Complete Validation Pipeline")
        print(f"{'='*70}")
        print(f"   - ImageNette Classification Models")
        print(f"   - Noise Detection Models")
        
        # Validate ImageNette classification models
        imagenette_results = self.validate_all_models()
        imagenette_summary = self.save_all_models_summary(imagenette_results) if imagenette_results else ""
        
        # Validate noise detection models
        noise_detection_results = self.validate_all_noise_detection_models()
        noise_detection_summary = self.save_noise_detection_summary(noise_detection_results) if noise_detection_results else ""
        
        # Save per-class results for ImageNette models
        per_class_files = []
        for result in imagenette_results:
            if result['success']:
                per_class_file = self.save_model_per_class_results(result)
                if per_class_file:
                    per_class_files.append(per_class_file)
        
        print(f"\n📁 Generated files:")
        print(f"   - 1 ImageNette classification summary")
        print(f"   - 1 noise detection summary")
        print(f"   - {len(per_class_files)} per-class ImageNette files")
        
        return {
            'imagenette_summary_csv': imagenette_summary,
            'noise_detection_summary_csv': noise_detection_summary,
            'per_class_csv_files': per_class_files
        }


if __name__ == "__main__":
    # Example usage
    validator = ImageNetteValidator()
    
    # ===== ImageNette Classification Validation =====
    # Run full ImageNette validation pipeline
    # files = validator.run_full_validation()
    
    # ===== Noise Detection Validation =====
    # Run noise detection validation pipeline
    noise_files = validator.run_noise_detection_validation()
    
    print(f"\n✅ Noise Detection Validation complete! Generated files:")
    
    # Display noise detection summary file
    if noise_files.get('noise_detection_summary_csv'):
        print(f"\n📊 Noise Detection Summary file:")
        print(f"   📄 {noise_files['noise_detection_summary_csv']}")
    
    # ===== Adversarial Models Validation =====
    # Example: Validate adversarial models
    # from attacks.attack_names import AttackNames
    # attack_names = ['FGSM', 'PGD', 'BIM', 'DeepFool']
    # adv_files = validator.run_adversarial_validation(
    #     attack_names=attack_names,
    #     batch_size=32
    # )
    
    # ===== Complete Validation (All Types) =====
    # Run complete validation for ImageNette, noise detection, and adversarial models
    # all_files = validator.run_complete_validation()
    
    # ===== Individual Model Validation Examples =====
    # Example: Validate specific ImageNette model
    # result = validator.validate_model('resnet18')
    # print(f"ResNet18 accuracy: {result['accuracy']:.4f}")
    
    # Example: Validate specific noise detection model
    # result = validator.validate_noise_detection_model('resnet18')
    # print(f"ResNet18 noise detector F1: {result['f1']:.2f}%")
    
    # Example: Validate specific adversarial model
    # result = validator.validate_adversarial_model('resnet18', ['FGSM', 'PGD'])
    # print(f"ResNet18 adversarial: Clean={result['clean_accuracy']:.2f}%, Adv={result['adversarial_accuracy']:.2f}%")
    
    # ===== Check Available Models =====
    # Example: Get available trained ImageNette models
    # available_models = validator._get_available_trained_models()
    # print(f"Available ImageNette models: {available_models}")
    
    # Example: Get available noise detection models
    # available_noise_models = validator._get_available_noise_detection_models()
    # print(f"Available noise detection models: {available_noise_models}")
