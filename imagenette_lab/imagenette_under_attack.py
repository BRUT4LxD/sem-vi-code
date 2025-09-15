import datetime
import time
import torch
import os
import csv
from typing import List
from torch.utils.data import DataLoader
from tqdm import tqdm
from attacks.attack import Attack
from attacks.system_under_attack import SystemUnderAttack
from attacks.white_box.fgsm import FGSM
from attacks.attack_factory import AttackFactory
from config.imagenette_classes import ImageNetteClasses
from data_eng.dataset_loader import load_imagenette
from data_eng.io import load_model_imagenette
from domain.attack.attack_result import AttackResult
from domain.model.model_names import ModelNames
from attacks.attack_names import AttackNames
from evaluation.metrics import Metrics
from shared.model_utils import ModelUtils

def attack_images_imagenette(attack: Attack, data_loader: DataLoader):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = attack.model

    model.eval()
    model.to(device)

    num_classes = len(ImageNetteClasses.get_classes())

    attack_results = []
    start_time = time.time()

    for images, labels in tqdm(data_loader):

        # 1. Remove missclassified images
        # 2. Attack the images
        # 3. Test the attacked images
        # 4. Collect the evaluation scores
        images, labels = images.to(device), labels.to(device)

        images, labels = ModelUtils.remove_missclassified_imagenette(attack.model, images, labels)
        if labels.numel() == 0:
            continue

        adv_images = attack(images, labels)
        
        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted_labels = torch.max(outputs.data, 1)
            for i in range(len(adv_images)):
                label = labels[i].item()
                predicted_label = predicted_labels[i].item()

                att_result = AttackResult(
                    actual=label,
                    predicted=predicted_label,
                    adv_image=adv_images[i].cpu(),
                    src_image=images[i].cpu(),
                    model_name=attack.model_name,
                    attack_name=attack.attack)

                attack_results.append(att_result)
    
    # 5. Calculate global evaluation scores
    if len(attack_results) == 0:
        print("Warning: No successful attacks generated")
        return None

    ev = Metrics.evaluate_attack_score(attack_results, num_classes)
    ev.set_after_attack(time.time() - start_time, len(attack_results))

    return ev


def run_attacks_on_models(
    model_names: List[str], 
    attack_names: List[str], 
    data_loader: DataLoader,
    device: str = 'cuda',
    save_results: bool = True,
    results_folder: str = "results/attacks/imagenette_models"
):
    """
    Run attacks on multiple models and save results to CSV files.
    
    Args:
        model_names: List of model names to test
        attack_names: List of attack names to use
        data_loader: DataLoader with test data
        device: Device to run on ('cuda' or 'cpu')
        save_results: Whether to save results to CSV files
        results_folder: Folder to save CSV files
    """
    
    # Create results directory if it doesn't exist
    if save_results:
        os.makedirs(results_folder, exist_ok=True)
    
    print(f"🚀 Starting comprehensive attack evaluation...")
    print(f"📊 Models: {len(model_names)}")
    print(f"⚔️  Attacks: {len(attack_names)}")
    print(f"📁 Results will be saved to: {results_folder}")
    print("=" * 80)
    
    # Initialize results storage
    all_results = {}
    
    for model_name in model_names:
        print(f"\n🔍 Loading model: {model_name}")
        
        # Load model
        model_path = f"./models/imagenette/{model_name}_advanced.pt"
        result = load_model_imagenette(model_path, model_name, device=device, verbose=False)
        
        if not result['success']:
            print(f"❌ Failed to load model {model_name}: {result['error']}")
            continue
            
        model = result['model']
        print(f"✅ Model {model_name} loaded successfully")
        
        # Store results for this model
        model_results = []
        
        # Run each attack on this model
        for attack_name in attack_names:
            print(f"  ⚔️  Running {attack_name} attack...")
            
            try:
                # Create attack
                attack = AttackFactory.get_attack(attack_name, model)
                
                # Run attack
                ev = attack_images_imagenette(attack, data_loader)
                
                if ev is not None:
                    result_dict = {}
                    result_dict['model_name'] = model_name  
                    result_dict['attack_name'] = attack_name
                    result_dict['acc'] = ev.acc
                    result_dict['prec'] = ev.prec
                    result_dict['rec'] = ev.rec
                    result_dict['f1'] = ev.f1
                    
                    distance_score = result_dict.pop('distance_score', {})
                    result_dict['l0_pixels'] = distance_score.get('l0_pixels', 0)
                    result_dict['l1'] = distance_score.get('l1', 0)
                    result_dict['l2'] = distance_score.get('l2', 0)
                    result_dict['linf'] = distance_score.get('linf', 0)
                    result_dict['power_mse'] = distance_score.get('power_mse', 0)

                    result_dict['time'] = ev.time
                    result_dict['n_samples'] = ev.n_samples
                    model_results.append(result_dict)
                    print(f"    ✅ {attack_name}: {ev.acc:.2f}% accuracy")
                else:
                    print(f"    ⚠️  {attack_name}: No results generated")
                    
            except Exception as e:
                print(f"    ❌ {attack_name} failed: {str(e)}")
                continue
        
        # Store results for this model
        all_results[model_name] = model_results
        
        # Save individual model results to CSV
        if save_results and model_results:
            save_model_results_to_csv(model_name, model_results, results_folder)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)
    
    for model_name, results in all_results.items():
        print(f"\n🔍 {model_name}:")
        if results:
            for result in results:
                attack_name = result['attack_name']
                accuracy = result['acc']
                print(f"  {attack_name:12s}: {accuracy:6.2f}% accuracy")
        else:
            print("  No successful results")
    
    return all_results


def save_model_results_to_csv(model_name: str, results: List[dict], results_folder: str):
    """
    Save results for a single model to CSV file.
    
    Args:
        model_name: Name of the model
        results: List of result dictionaries
        results_folder: Folder to save the CSV file
    """
    
    if not results:
        return
    
    # Create CSV filename
    csv_filename = f"{model_name}.csv"
    csv_path = os.path.join(results_folder, csv_filename)
    
    # Get fieldnames from first result (excluding model_name and conf_matrix)
    fieldnames = [key for key in results[0].keys() if key not in ['model_name', 'conf_matrix']]
    # Put model_name first
    fieldnames = ['model_name'] + fieldnames
    
    # Write CSV file
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Write data rows (excluding conf_matrix)
        for result in results:
            filtered_result = {key: value for key, value in result.items() if key != 'conf_matrix'}
            writer.writerow(filtered_result)
    
    print(f"    💾 Saved results to: {csv_path}")


def run_simple_example():
    """
    Simple example with just 2 models and 2 attacks for quick testing.
    """
    print("🧪 Running simple example...")
    
    # Simple test with fewer models and attacks
    model_names = [
        ModelNames().resnet18,
        ModelNames().vgg16
    ]
    
    attack_names = [
        AttackNames().FGSM,
        AttackNames().PGD
    ]
    
    # Get test data (very small for quick testing)
    _, test_loader = load_imagenette(batch_size=2, test_subset_size=50)
    
    # Run comprehensive attack evaluation
    results = run_attacks_on_models(
        model_names=model_names,
        attack_names=attack_names,
        data_loader=test_loader,
        device='cuda',
        save_results=True,
        results_folder="results/attacks/imagenette_models"
    )
    
    return results


# Example usage of the comprehensive attack evaluation
if __name__ == "__main__":
    
    # Run simple example for testing
    results = run_simple_example()
    
    print(f"\n🎉 Simple example completed!")
    print(f"📁 Results saved to: results/attacks/imagenette_models/")
    
    # Uncomment below for full evaluation with more models and attacks:
    # 
    # # Define models and attacks to test
    # model_names = [
    #     ModelNames().resnet18,
    #     ModelNames().vgg16,
    #     ModelNames().densenet121,
    #     ModelNames().mobilenet_v2,
    #     ModelNames().efficientnet_b0
    # ]
    # 
    # attack_names = [
    #     AttackNames().FGSM,
    #     AttackNames().PGD,
    #     AttackNames().BIM,
    #     AttackNames().CW,
    #     AttackNames().DeepFool,
    #     AttackNames().APGD
    # ]
    # 
    # # Get test data
    # _, test_loader = load_imagenette(batch_size=8, test_subset_size=200)
    # 
    # # Run comprehensive attack evaluation
    # results = run_attacks_on_models(
    #     model_names=model_names,
    #     attack_names=attack_names,
    #     data_loader=test_loader,
    #     device='cuda',
    #     save_results=True,
    #     results_folder="results/attacks/imagenette_models"
    # )
    
    # Example: Run single attack on single model (original functionality)
    # model_name = ModelNames().resnet18
    # model_path = f"./models/imagenette/{model_name}_advanced.pt"
    # result = load_model_imagenette(model_path, model_name, device='cuda')
    # 
    # if not result['success']:
    #     print(f"❌ Failed to load model: {result['error']}")
    #     exit()
    # 
    # model = result['model']
    # print(f"✅ Loaded model: {result['model_name']}")
    # 
    # # Create FGSM attack
    # attack = FGSM(model)
    # 
    # # Get test data (small subset for quick testing)
    # _, test_loader = load_imagenette(batch_size=16, test_subset_size=200)
    # 
    # attack_images_imagenette(attack, test_loader, save_results=False)