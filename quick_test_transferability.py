#!/usr/bin/env python3
"""
Quick test script for transferability attacks - minimal setup for rapid testing.
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from imagenette_lab.imagenette_transferability_attacks import (
    imagenette_transferability_model2model_in_memory,
    imagenette_transferability_attack2model_in_memory
)
from domain.model.model_names import ModelNames
from attacks.attack_names import AttackNames


def quick_test():
    """
    Quick test with minimal setup for rapid verification.
    """
    print("🚀 Quick Transferability Test")
    print("=" * 40)
    
    # Minimal test configuration
    model_names = [ModelNames().resnet18, ModelNames().densenet121]
    attack_names = [AttackNames().FGSM]
    images_per_attack = 5
    results_folder = "quick_test_results"
    
    print(f"📊 Models: {model_names}")
    print(f"🎯 Attack: {attack_names[0]}")
    print(f"🖼️ Images: {images_per_attack}")
    
    try:
        # Test model-to-model transferability
        print("\n🔄 Testing Model-to-Model...")
        results = imagenette_transferability_model2model_in_memory(
            model_names=model_names,
            attack_names=attack_names,
            images_per_attack=images_per_attack,
            batch_size=1,
            results_folder=results_folder
        )
        
        print(f"✅ Results: {len(results)} transferability tests")
        for result in results:
            print(f"  {result.source_model} → {result.target_model}: "
                  f"{result.transfer_success}/{result.source_success} "
                  f"({result.transfer_rate:.1%})")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n🎉 Quick test passed!")
    else:
        print("\n💥 Quick test failed!")
