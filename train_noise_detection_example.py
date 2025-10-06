#!/usr/bin/env python3
"""
Example script for training adversarial noise detection models.

This script demonstrates how to train binary classifiers to detect
adversarial examples in ImageNette dataset.
"""

import torch
from config.imagenet_models import ImageNetModels
from data_eng.dataset_loader import load_attacked_imagenette
from domain.model.model_names import ModelNames
from training.train import Training


def train_noise_detector_resnet18():
    """
    Train ResNet18 for adversarial noise detection.
    """
    print("🎯 Training ResNet18 for Adversarial Noise Detection")
    print("=" * 60)
    
    # 1. Load dataset
    print("\n📁 Loading attacked ImageNette dataset...")
    train_loader, test_loader = load_attacked_imagenette(
        attacked_images_folder="data/attacks/imagenette_models",
        clean_images_folder="./data/imagenette/val",
        test_images_per_attack=2,
        batch_size=32,
        shuffle=True
    )
    
    # 2. Setup model
    print("\n🔧 Setting up model...")
    model_name = ModelNames().resnet18
    model = ImageNetModels.get_model(model_name)
    
    # 3. Train
    print("\n🏋️ Starting training...")
    training_state = Training.train_imagenette_noise_detection(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=0.001,
        num_epochs=20,
        device='cuda',
        save_model_path=f"./models/binary/{model_name}_noise_detector.pt",
        model_name=model_name,
        setup_model=True,
        validation_frequency=1,
        early_stopping_patience=7,
        min_delta=0.001,
        scheduler_type='plateau',
        gradient_clip_norm=1.0,
        weight_decay=0.0001,
        verbose=True
    )
    
    print(f"\n✅ Training completed!")
    print(f"📊 Best validation accuracy: {training_state['best_val_accuracy']:.2f}%")
    print(f"📊 Best F1 score: {training_state['best_f1']:.2f}%")
    
    return training_state


def train_noise_detector_all_models():
    """
    Train multiple models for adversarial noise detection.
    """
    print("🎯 Training Multiple Models for Adversarial Noise Detection")
    print("=" * 60)
    
    # Model names to train
    model_names = [
        ModelNames().resnet18,
        ModelNames().densenet121,
        ModelNames().mobilenet_v2,
        ModelNames().efficientnet_b0
    ]
    
    # Load dataset once (reuse for all models)
    print("\n📁 Loading attacked ImageNette dataset...")
    train_loader, test_loader = load_attacked_imagenette(
        attacked_images_folder="data/attacks/imagenette_models",
        clean_images_folder="./data/imagenette/val",
        test_images_per_attack=2,
        batch_size=32,
        shuffle=True
    )
    
    results = {}
    
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"🔧 Training {model_name}...")
        print(f"{'='*60}")
        
        try:
            # Setup model
            model = ImageNetModels.get_model(model_name)
            
            # Train
            training_state = Training.train_imagenette_noise_detection(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                learning_rate=0.001,
                num_epochs=20,
                device='cuda',
                save_model_path=f"./models/binary/{model_name}_noise_detector.pt",
                model_name=model_name,
                setup_model=True,
                validation_frequency=1,
                early_stopping_patience=7,
                scheduler_type='plateau',
                weight_decay=0.0001,
                verbose=True
            )
            
            results[model_name] = training_state
            print(f"✅ {model_name} training completed!")
            
        except Exception as e:
            print(f"❌ {model_name} training failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary of all models
    print(f"\n{'='*60}")
    print(f"📊 Training Summary for All Models")
    print(f"{'='*60}")
    
    for model_name, state in results.items():
        print(f"\n{model_name}:")
        print(f"   Best Val Acc: {state['best_val_accuracy']:.2f}%")
        print(f"   Precision: {state['best_precision']:.2f}%")
        print(f"   Recall: {state['best_recall']:.2f}%")
        print(f"   F1 Score: {state['best_f1']:.2f}%")
        print(f"   Training Time: {state['total_training_time']:.2f}s")
    
    return results


def quick_test_training():
    """
    Quick test with minimal epochs to verify training pipeline.
    """
    print("🧪 Quick Test Training")
    print("=" * 60)
    
    # Load dataset
    print("\n📁 Loading dataset...")
    train_loader, test_loader = load_attacked_imagenette(
        attacked_images_folder="data/attacks/imagenette_models",
        clean_images_folder="./data/imagenette/val",
        test_images_per_attack=2,
        batch_size=16,
        shuffle=True
    )
    
    # Setup model
    print("\n🔧 Setting up model...")
    model_name = ModelNames().resnet18
    model = ImageNetModels.get_model(model_name)
    
    # Train for just 2 epochs as a test
    print("\n🏋️ Starting quick test (2 epochs)...")
    training_state = Training.train_imagenette_noise_detection(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=0.001,
        num_epochs=2,  # Quick test
        device='cuda',
        save_model_path=None,  # Don't save
        model_name=f"{model_name}_test",
        setup_model=True,
        verbose=True
    )
    
    print(f"\n✅ Quick test completed!")
    print(f"📊 Final validation accuracy: {training_state['best_val_accuracy']:.2f}%")
    
    return training_state


def train_with_custom_config():
    """
    Example with custom training configuration.
    """
    print("🎯 Training with Custom Configuration")
    print("=" * 60)
    
    # Load dataset
    train_loader, test_loader = load_attacked_imagenette(
        attacked_images_folder="data/attacks/imagenette_models",
        clean_images_folder="./data/imagenette/val",
        test_images_per_attack=2,
        batch_size=64,  # Larger batch size
        shuffle=True
    )
    
    # Setup model
    model_name = ModelNames().efficientnet_b0
    model = ImageNetModels.get_model(model_name)
    
    # Custom scheduler parameters
    custom_scheduler_params = {
        'factor': 0.5,
        'patience': 5
    }
    
    # Train with custom configuration
    training_state = Training.train_imagenette_noise_detection(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=0.0001,  # Lower learning rate
        num_epochs=30,  # More epochs
        device='cuda',
        save_model_path=f"./models/binary/{model_name}_noise_detector_custom.pt",
        model_name=model_name,
        setup_model=True,
        validation_frequency=1,
        early_stopping_patience=10,  # More patience
        min_delta=0.0005,
        scheduler_type='plateau',
        scheduler_params=custom_scheduler_params,
        gradient_clip_norm=1.0,  # Enable gradient clipping
        weight_decay=0.001,  # Higher weight decay
        verbose=True
    )
    
    print(f"\n✅ Custom training completed!")
    return training_state


if __name__ == "__main__":
    import sys
    
    # Choose which example to run
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("Usage:")
        print("  python train_noise_detection_example.py quick    - Quick test (2 epochs)")
        print("  python train_noise_detection_example.py single   - Train ResNet18")
        print("  python train_noise_detection_example.py all      - Train all models")
        print("  python train_noise_detection_example.py custom   - Train with custom config")
        mode = 'quick'
    
    if mode == 'quick':
        quick_test_training()
    elif mode == 'single':
        train_noise_detector_resnet18()
    elif mode == 'all':
        train_noise_detector_all_models()
    elif mode == 'custom':
        train_with_custom_config()
    else:
        print(f"Unknown mode: {mode}")

