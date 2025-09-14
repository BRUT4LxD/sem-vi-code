#!/usr/bin/env python3
"""
Example usage of ImageNetteValidator.

This script demonstrates how to use the ImageNetteValidator to validate
all trained models and generate comprehensive reports.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imagenette_lab.imagenette_validator import ImageNetteValidator


def main():
    """Main function demonstrating validator usage."""
    
    print("🚀 ImageNette Validator Example Usage")
    print("=" * 50)
    
    # Initialize validator
    validator = ImageNetteValidator(
        models_dir='./models/imagenette',
        results_dir='./results/imagenette_trained_models',
        device='auto'  # Will use CUDA if available, otherwise CPU
    )
    
    # Check available trained models
    print("\n📋 Checking available trained models...")
    available_models = validator._get_available_trained_models()
    print(f"Found {len(available_models)} trained models:")
    for model_name in available_models:
        print(f"   ✅ {model_name}")
    
    if not available_models:
        print("❌ No trained models found!")
        print("   Make sure you have trained models in ./models/imagenette/")
        print("   Expected format: {model_name}_advanced.pt")
        return
    
    # Run full validation pipeline
    print(f"\n🔍 Running full validation pipeline...")
    files = validator.run_full_validation()
    
    # Display results
    print(f"\n✅ Validation complete! Generated files:")
    
    # Display summary files
    if files.get('summary_csv_files'):
        print(f"\n📊 Summary files ({len(files['summary_csv_files'])}):")
        for file_path in files['summary_csv_files']:
            print(f"   📄 {file_path}")
    
    # Display per-class files
    if files.get('per_class_csv_files'):
        print(f"\n📊 Per-class files ({len(files['per_class_csv_files'])}):")
        for file_path in files['per_class_csv_files']:
            print(f"   📄 {file_path}")
    
    # Display report file
    if files.get('report_txt'):
        print(f"\n📋 Overall report:")
        print(f"   📄 {files['report_txt']}")
    
    # Example: Validate specific model
    print(f"\n🎯 Example: Validate specific model...")
    if available_models:
        model_name = available_models[0]
        result = validator.validate_model(model_name)
        if result['success']:
            print(f"   {model_name} Overall Accuracy: {result['overall_accuracy']:.4f}")
            print(f"   {model_name} Overall F1: {result['overall_f1']:.4f}")
        else:
            print(f"   ❌ Failed to validate {model_name}: {result['error']}")


if __name__ == "__main__":
    main()
