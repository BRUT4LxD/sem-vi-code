#!/usr/bin/env python3
"""
Test script for in-memory transferability attack implementations.

This script demonstrates how to use the transferability attack functions
with various configurations and provides examples for testing.
"""

import sys
import os
import torch

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from imagenette_lab.imagenette_transferability_attacks import (
    imagenette_transferability_model2model_in_memory,
    imagenette_transferability_attack2model_in_memory,
    run_all_transferability_experiments
)
from domain.model.model_names import ModelNames
from attacks.attack_names import AttackNames


def models_from_names(architecture_names, models_root: str = "./models/imagenette"):
    """Build ``List[Tuple[str, str]]`` as (name, path) for transferability APIs."""
    return [
        (n, os.path.normpath(os.path.join(models_root, f"{n}_advanced.pt")))
        for n in architecture_names
    ]


def test_model2model_transferability():
    """
    Test in-memory model-to-model transferability analysis.
    """
    print("🧪 Testing Model-to-Model Transferability")
    print("=" * 50)
    
    # Define test parameters
    model_names = [ModelNames().resnet18, ModelNames().densenet121]
    models = models_from_names(model_names)
    attack_names = [AttackNames().FGSM, AttackNames().PGD]
    images_per_attack = 10  # Small number for testing
    results_folder = "test_results/transferability"
    
    print(f"📊 Models: {model_names}")
    print(f"🎯 Attacks: {attack_names}")
    print(f"🖼️ Images per attack: {images_per_attack}")
    print(f"📁 Results folder: {results_folder}")
    
    try:
        # Run model-to-model transferability analysis
        results = imagenette_transferability_model2model_in_memory(
            models=models,
            attack_names=attack_names,
            images_per_attack=images_per_attack,
            batch_size=1,
            results_folder=results_folder
        )
        
        print(f"\n✅ Model-to-Model Transferability Test Completed!")
        print(f"📊 Total results: {len(results)}")
        
        # Print summary of results
        for result in results:
            print(f"  {result.source_model} → {result.target_model} ({result.attack_name}): "
                  f"{result.transfer_success}/{result.total_successful_attacks} "
                  f"({result.transfer_rate:.2%})")
        
        return results
        
    except Exception as e:
        print(f"❌ Model-to-Model Transferability Test Failed: {e}")
        return None


def test_attack2model_transferability():
    """
    Test in-memory attack-to-model transferability analysis.
    """
    print("\n🧪 Testing Attack-to-Model Transferability")
    print("=" * 50)
    
    # Define test parameters
    model_names = [ModelNames().resnet18, ModelNames().densenet121]
    models = models_from_names(model_names)
    attack_names = [AttackNames().FGSM, AttackNames().PGD]
    images_per_attack = 10  # Small number for testing
    results_folder = "test_results/transferability"
    
    print(f"📊 Models: {model_names}")
    print(f"🎯 Attacks: {attack_names}")
    print(f"🖼️ Images per attack: {images_per_attack}")
    print(f"📁 Results folder: {results_folder}")
    
    try:
        # Run attack-to-model transferability analysis
        results = imagenette_transferability_attack2model_in_memory(
            models=models,
            attack_names=attack_names,
            images_per_attack=images_per_attack,
            batch_size=1,
            results_folder=results_folder
        )
        
        print(f"\n✅ Attack-to-Model Transferability Test Completed!")
        print(f"📊 Total results: {len(results)}")
        
        # Print summary of results
        for result in results:
            print(f"  {result.attack_name} → {result.target_model}: "
                  f"{result.transfer_success}/{result.total_successful_attacks} "
                  f"({result.transfer_rate:.2%})")
        
        return results
        
    except Exception as e:
        print(f"❌ Attack-to-Model Transferability Test Failed: {e}")
        return None


def test_comprehensive_experiment():
    """
    Test comprehensive transferability experiment (in-memory only).
    """
    print("\n🧪 Testing Comprehensive Transferability Experiment")
    print("=" * 50)
    
    # Define test parameters
    model_names = [ModelNames().resnet18, ModelNames().densenet121]
    models = models_from_names(model_names)
    attack_names = [AttackNames().FGSM, AttackNames().PGD]
    images_per_attack = 5  # Very small number for quick testing
    results_folder = "test_results/transferability"
    
    print(f"📊 Models: {model_names}")
    print(f"🎯 Attacks: {attack_names}")
    print(f"🖼️ Images per attack: {images_per_attack}")
    print(f"📁 Results folder: {results_folder}")
    
    try:
        # Run comprehensive experiment (only in-memory methods)
        all_results = {}
        
        # Test model-to-model transferability
        print("\n🔄 Running Model-to-Model Transferability...")
        model2model_results = imagenette_transferability_model2model_in_memory(
            models=models,
            attack_names=attack_names,
            images_per_attack=images_per_attack,
            batch_size=1,
            results_folder=results_folder
        )
        all_results["Model-to-Model"] = model2model_results
        
        # Test attack-to-model transferability
        print("\n🔄 Running Attack-to-Model Transferability...")
        attack2model_results = imagenette_transferability_attack2model_in_memory(
            models=models,
            attack_names=attack_names,
            images_per_attack=images_per_attack,
            batch_size=1,
            results_folder=results_folder
        )
        all_results["Attack-to-Model"] = attack2model_results
        
        print(f"\n✅ Comprehensive Transferability Test Completed!")
        
        # Print summary of all results
        for exp_name, results in all_results.items():
            print(f"\n📊 {exp_name} Results ({len(results)} total):")
            for result in results:
                if "Model-to-Model" in exp_name:
                    print(f"  {result.source_model} → {result.target_model} ({result.attack_name}): "
                          f"{result.transfer_success}/{result.total_successful_attacks} "
                          f"({result.transfer_rate:.2%})")
                else:
                    print(f"  {result.attack_name} → {result.target_model}: "
                          f"{result.transfer_success}/{result.total_successful_attacks} "
                          f"({result.transfer_rate:.2%})")
        
        return all_results
        
    except Exception as e:
        print(f"❌ Comprehensive Transferability Test Failed: {e}")
        return None


def test_single_model_single_attack():
    """
    Test with single model and single attack for debugging.
    """
    print("\n🧪 Testing Single Model Single Attack (Debug Mode)")
    print("=" * 50)
    
    # Define minimal test parameters
    model_names_one = [ModelNames().resnet18]
    models_one = models_from_names(model_names_one)
    model_names_two = [ModelNames().resnet18, ModelNames().densenet121]
    models_two = models_from_names(model_names_two)
    attack_names = [AttackNames().FGSM]
    images_per_attack = 3  # Very small for debugging
    results_folder = "test_results/transferability_debug"
    
    print(f"📊 Models (m2m): {model_names_one} | (a2m): {model_names_two}")
    print(f"🎯 Attacks: {attack_names}")
    print(f"🖼️ Images per attack: {images_per_attack}")
    print(f"📁 Results folder: {results_folder}")
    
    try:
        # Test model-to-model (should have no results since only one model)
        print("\n🔄 Testing Model-to-Model (should be empty)...")
        model2model_results = imagenette_transferability_model2model_in_memory(
            models=models_one,
            attack_names=attack_names,
            images_per_attack=images_per_attack,
            batch_size=1,
            results_folder=results_folder
        )
        
        print(f"Model-to-Model results: {len(model2model_results)} (expected: 0)")
        
        # Attack-to-model requires at least two checkpoints
        print("\n🔄 Testing Attack-to-Model...")
        attack2model_results = imagenette_transferability_attack2model_in_memory(
            models=models_two,
            attack_names=attack_names,
            images_per_attack=images_per_attack,
            batch_size=1,
            results_folder=results_folder
        )
        
        print(f"Attack-to-Model results: {len(attack2model_results)} (expected: >=1)")
        
        # Print results
        for result in attack2model_results:
            print(f"  {result.attack_name} → {result.target_model}: "
                  f"{result.transfer_success}/{result.total_successful_attacks} "
                  f"({result.transfer_rate:.2%})")
        
        print(f"\n✅ Single Model Single Attack Test Completed!")
        return attack2model_results
        
    except Exception as e:
        print(f"❌ Single Model Single Attack Test Failed: {e}")
        return None


def test_error_handling():
    """
    Test error handling with invalid parameters.
    """
    print("\n🧪 Testing Error Handling")
    print("=" * 50)
    
    try:
        # Missing checkpoint file
        print("🔄 Testing with missing .pt file...")
        try:
            imagenette_transferability_model2model_in_memory(
                models=[("resnet18", "./this_checkpoint_does_not_exist_xxx.pt")],
                attack_names=[AttackNames().FGSM],
                images_per_attack=5,
                results_folder="test_results/error_test",
            )
        except FileNotFoundError:
            print("  Caught FileNotFoundError as expected")

        # Test with invalid attack names
        print("🔄 Testing with invalid attack names...")
        invalid_results2 = imagenette_transferability_model2model_in_memory(
            models=models_from_names([ModelNames().resnet18, ModelNames().densenet121]),
            attack_names=["invalid_attack"],
            images_per_attack=5,
            results_folder="test_results/error_test",
        )
        print(f"Results with invalid attacks: {len(invalid_results2)} (expected: 0)")
        
        print("✅ Error Handling Test Completed!")
        
    except Exception as e:
        print(f"❌ Error Handling Test Failed: {e}")


def run_performance_test():
    """
    Test performance with different batch sizes.
    """
    print("\n🧪 Testing Performance with Different Batch Sizes")
    print("=" * 50)
    
    model_names = [ModelNames().resnet18, ModelNames().densenet121]
    models = models_from_names(model_names)
    attack_names = [AttackNames().FGSM]
    images_per_attack = 10
    results_folder = "test_results/performance_test"
    
    batch_sizes = [1, 2, 4]
    
    for batch_size in batch_sizes:
        print(f"\n🔄 Testing with batch_size={batch_size}...")
        
        try:
            import time
            start_time = time.time()
            
            results = imagenette_transferability_model2model_in_memory(
                models=models,
                attack_names=attack_names,
                images_per_attack=images_per_attack,
                batch_size=batch_size,
                results_folder=results_folder
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"  Batch size {batch_size}: {len(results)} results in {duration:.2f}s")
            
        except Exception as e:
            print(f"  Batch size {batch_size}: Failed - {e}")
    
    print("✅ Performance Test Completed!")


def main():
    """
    Main function to run all tests.
    """
    print("🚀 Starting Transferability Attack Tests")
    print("=" * 60)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Device: {device}")
    
    # Create test results directory
    os.makedirs("test_results", exist_ok=True)
    
    # Run tests in order of complexity
    try:
        # 1. Single model single attack test (simplest)
        print("\n" + "="*60)
        test_single_model_single_attack()
        
        # 2. Model-to-model transferability test
        print("\n" + "="*60)
        test_model2model_transferability()
        
        # 3. Attack-to-model transferability test
        print("\n" + "="*60)
        test_attack2model_transferability()
        
        # 4. Comprehensive experiment test
        print("\n" + "="*60)
        test_comprehensive_experiment()
        
        # 5. Error handling test
        print("\n" + "="*60)
        test_error_handling()
        
        # 6. Performance test
        print("\n" + "="*60)
        run_performance_test()
        
        print("\n🎉 All Transferability Attack Tests Completed Successfully!")
        print("📁 Check the 'test_results/' directory for generated CSV files and logs.")
        
    except Exception as e:
        print(f"\n❌ Test Suite Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
