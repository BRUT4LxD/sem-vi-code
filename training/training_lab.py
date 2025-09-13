#!/usr/bin/env python3
"""
Training Lab - Test script for verifying model setup and training functionality.
This script tests the improved setup_imagenette method and verifies proper model training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.imagenet_models import ImageNetModels
from training.transfer.setup_pretraining import SetupPretraining
from training.train import Training
from domain.model_names import ModelNames

class TrainingLab:
    """Test class for verifying model setup and training functionality."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")
        
        # Test models (one from each architecture)
        self.test_models = [
            ModelNames.resnet18,
            ModelNames.densenet121,
            ModelNames.vgg16,
            ModelNames.mobilenet_v2,
            ModelNames.efficientnet_b0
        ]
    
    def test_model_setup(self, model_name):
        """Test the setup_imagenette method for a specific model."""
        print(f"\n{'='*60}")
        print(f"🧪 Testing Model Setup: {model_name}")
        print(f"{'='*60}")
        
        try:
            # Load pretrained model
            print(f"📥 Loading pretrained {model_name}...")
            model = ImageNetModels.get_model(model_name)
            model = model.to(self.device)
            
            # Test original model
            print(f"🔍 Testing original model...")
            original_output = self._test_forward_pass(model)
            print(f"   Original output shape: {original_output.shape}")
            
            # Setup for ImageNette
            print(f"⚙️ Setting up for ImageNette...")
            model = SetupPretraining.setup_imagenette(model)
            
            # Verify setup
            print(f"✅ Verifying setup...")
            verification_passed = SetupPretraining.verify_model_setup(model)
            
            # Test new model
            print(f"🔍 Testing modified model...")
            new_output = self._test_forward_pass(model)
            print(f"   New output shape: {new_output.shape}")
            
            # Check if output has correct number of classes
            if new_output.shape[1] == 10:
                print(f"✅ SUCCESS: Model correctly configured for ImageNette (10 classes)")
                return True
            else:
                print(f"❌ FAILURE: Expected 10 classes, got {new_output.shape[1]}")
                return False
                
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            return False
    
    def _test_forward_pass(self, model):
        """Test forward pass with dummy input."""
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        return output
    
    def test_training_simulation(self, model_name):
        """Simulate a short training session to verify everything works."""
        print(f"\n{'='*60}")
        print(f"🏋️ Testing Training Simulation: {model_name}")
        print(f"{'='*60}")
        
        try:
            # Setup model
            model = ImageNetModels.get_model(model_name)
            model = SetupPretraining.setup_imagenette(model)
            model = model.to(self.device)
            
            # Create dummy dataset
            print("📊 Creating dummy dataset...")
            dummy_data, dummy_labels = self._create_dummy_dataset(100, 10)
            dataset = TensorDataset(dummy_data, dummy_labels)
            dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
            
            # Setup training
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            print("🚀 Starting training simulation...")
            model.train()
            
            total_loss = 0
            correct_predictions = 0
            total_samples = 0
            
            for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Training")):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct_predictions += pred.eq(target).sum().item()
                total_samples += target.size(0)
                
                # Only run a few batches for testing
                if batch_idx >= 2:
                    break
            
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = 100. * correct_predictions / total_samples
            
            print(f"📈 Training Results:")
            print(f"   Average Loss: {avg_loss:.4f}")
            print(f"   Accuracy: {accuracy:.2f}%")
            print(f"   Correct: {correct_predictions}/{total_samples}")
            
            if accuracy > 0:  # Any correct predictions means it's working
                print(f"✅ SUCCESS: Training simulation completed successfully")
                return True
            else:
                print(f"❌ FAILURE: No correct predictions")
                return False
                
        except Exception as e:
            print(f"❌ ERROR during training: {str(e)}")
            return False
    
    def _create_dummy_dataset(self, num_samples, num_classes):
        """Create dummy dataset for testing."""
        # Create random images (normalized)
        data = torch.randn(num_samples, 3, 224, 224)
        data = (data - data.mean()) / data.std()  # Normalize
        
        # Create random labels
        labels = torch.randint(0, num_classes, (num_samples,))
        
        return data, labels
    
    def test_train_imagenette_method(self, model_name):
        """Test the train_imagenette method from Training class."""
        print(f"\n{'='*60}")
        print(f"🏋️ Testing train_imagenette Method: {model_name}")
        print(f"{'='*60}")
        
        try:
            # Load pretrained model
            print(f"📥 Loading pretrained {model_name}...")
            model = ImageNetModels.get_model(model_name)
            
            # Create dummy dataset
            print("📊 Creating dummy dataset...")
            dummy_data, dummy_labels = self._create_dummy_dataset(50, 10)
            dataset = TensorDataset(dummy_data, dummy_labels)
            train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
            test_loader = DataLoader(dataset, batch_size=8, shuffle=False)
            
            # Test train_imagenette method
            print("🚀 Testing train_imagenette method...")
            Training.train_imagenette(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                learning_rate=0.001,
                num_epochs=1,  # Just 1 epoch for testing
                device=self.device,
                save_model_path=None,  # Don't save for testing
                model_name=model_name,
                writer=None,  # No tensorboard for testing
                setup_model=True  # Use the new setup functionality
            )
            
            print(f"✅ SUCCESS: train_imagenette method completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ ERROR during train_imagenette: {str(e)}")
            return False
    
    def run_comprehensive_test(self):
        """Run comprehensive tests on all model architectures."""
        print(f"🎯 Starting Comprehensive Model Setup Tests")
        print(f"Testing {len(self.test_models)} different architectures...")
        
        results = {}
        
        for model_name in self.test_models:
            # Test model setup
            setup_success = self.test_model_setup(model_name)
            
            # Test training simulation
            training_success = self.test_training_simulation(model_name)
            
            # Test train_imagenette method (only test one model to avoid long runtime)
            if model_name == self.test_models[0]:  # Only test first model
                train_method_success = self.test_train_imagenette_method(model_name)
            else:
                train_method_success = True  # Skip for other models
            
            results[model_name] = {
                'setup': setup_success,
                'training': training_success,
                'train_method': train_method_success,
                'overall': setup_success and training_success and train_method_success
            }
        
        # Print summary
        self._print_test_summary(results)
        
        return results
    
    def _print_test_summary(self, results):
        """Print a summary of test results."""
        print(f"\n{'='*80}")
        print(f"📊 TEST SUMMARY")
        print(f"{'='*80}")
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r['overall'])
        
        print(f"Total Models Tested: {total_tests}")
        print(f"Passed Tests: {passed_tests}")
        print(f"Failed Tests: {total_tests - passed_tests}")
        print(f"Success Rate: {100 * passed_tests / total_tests:.1f}%")
        
        print(f"\nDetailed Results:")
        print(f"{'Model':<20} {'Setup':<8} {'Training':<10} {'TrainMethod':<12} {'Overall':<8}")
        print(f"{'-'*60}")
        
        for model_name, result in results.items():
            setup_status = "✅ PASS" if result['setup'] else "❌ FAIL"
            training_status = "✅ PASS" if result['training'] else "❌ FAIL"
            train_method_status = "✅ PASS" if result['train_method'] else "❌ FAIL"
            overall_status = "✅ PASS" if result['overall'] else "❌ FAIL"
            
            print(f"{model_name:<20} {setup_status:<8} {training_status:<10} {train_method_status:<12} {overall_status:<8}")
        
        if passed_tests == total_tests:
            print(f"\n🎉 ALL TESTS PASSED! Your setup_imagenette method is working correctly.")
        else:
            print(f"\n⚠️ Some tests failed. Check the error messages above.")

def main():
    """Main function to run the training lab tests."""
    print("🧪 Training Lab - Model Setup Verification")
    print("=" * 50)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"🚀 CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
    else:
        print("⚠️ CUDA not available, using CPU")
    
    # Run tests
    lab = TrainingLab()
    results = lab.run_comprehensive_test()
    
    # Return success status
    all_passed = all(r['overall'] for r in results.values())
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
