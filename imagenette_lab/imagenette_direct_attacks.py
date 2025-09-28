import datetime
import time
import torch
import os
import csv
import traceback
from typing import List
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import ToPILImage
from attacks import get_all_attacks
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

def attack_images_imagenette(attack: Attack, data_loader: DataLoader, successfully_attacked_images_folder: str = ""):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = attack.model

    model.eval()
    model.to(device)

    num_classes = len(ImageNetteClasses.get_classes())

    attack_results = []
    start_time = time.time()

    clean_correct = 0
    clean_total = 0

    for images, labels in tqdm(data_loader):

        # 1. Remove missclassified images
        # 2. Attack the images
        # 3. Test the attacked images
        # 4. Collect the evaluation scores
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        
        # Calculate clean accuracy
        clean_total += labels.size(0)
        clean_correct += (predictions == labels).sum().item()

        # Remove missclassified images
        images, labels = images[predictions == labels], labels[predictions == labels]

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

    clean_accuracy = 100.0 * clean_correct / clean_total if clean_total > 0 else 0.0
    ev = Metrics.evaluate_attack_score(attack_results, num_classes, clean_accuracy=clean_accuracy)
    ev.set_after_attack(time.time() - start_time, len(attack_results))

    return ev


def attack_and_save_images(attack: Attack, data_loader: DataLoader, images_per_attack = 100, successfully_attacked_images_folder: str = ""):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = attack.model

    # create folder if doesn't exist
    if not os.path.exists(successfully_attacked_images_folder):
        os.makedirs(successfully_attacked_images_folder, exist_ok=True)

    model.eval()
    model.to(device)
    attacked_images_count = 0

    for images, labels in tqdm(data_loader):
        if attacked_images_count >= images_per_attack:
            break

        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        
        # Remove missclassified images
        images, labels = images[predictions == labels], labels[predictions == labels]

        if labels.numel() == 0:
            continue

        adv_images = attack(images, labels)
        
        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted_labels = torch.max(outputs.data, 1)
            for i in range(len(adv_images)):
                label = labels[i].item()
                predicted_label = predicted_labels[i].item()

                if predicted_label == label:
                    continue

                attacked_images_count += 1

                if attacked_images_count >= images_per_attack:
                    break

                # Path: successfully_attacked_images_folder/model_name/attack_name/label/timestamp.png
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Create directory structure
                save_dir = os.path.join(successfully_attacked_images_folder, attack.model_name, attack.attack, str(label))
                os.makedirs(save_dir, exist_ok=True)
                
                attcked_image_save_path = os.path.join(save_dir, f"{timestamp}.png")
                source_image_save_path = os.path.join(save_dir, f"src_{timestamp}.png")
                
                # Convert tensors to PIL Images and save as actual PNG files
                to_pil = ToPILImage()
                
                # Convert adversarial image tensor to PIL Image and save
                adv_pil = to_pil(adv_images[i].cpu())
                adv_pil.save(attcked_image_save_path)
                
                # Convert source image tensor to PIL Image and save
                src_pil = to_pil(images[i].cpu())
                src_pil.save(source_image_save_path)

def attack_and_save_images_multiple(model_names: List[str], attack_names: List[str], images_per_attack = 100, successfully_attacked_images_folder: str = ""):
    for model_name in model_names:
        model_path = f"./models/imagenette/{model_name}_advanced.pt"
        result = load_model_imagenette(model_path, model_name, device='cuda')
        model = result['model']
        for attack_name in attack_names:
            print(f"🔍 Running {attack_name} attack on {model_name}...")
            _, test_loader = load_imagenette(batch_size=1, test_subset_size=-1)
            attack = AttackFactory.get_attack(attack_name, model)
            try:
                attack_and_save_images(attack, test_loader, images_per_attack=images_per_attack, successfully_attacked_images_folder=successfully_attacked_images_folder)
            except Exception as e:
                print(f"❌ {attack_name} attack on {model_name} failed: {str(e)}")
                save_failure_log(model_name, attack_name, e, f"{successfully_attacked_images_folder}/{model_name}/{attack_name}")
                continue
            print(f"✅ {attack_name} attack on {model_name} completed!")

def save_failure_log(model_name: str, attack_name: str, exception: Exception, results_folder: str):
    """
    Save failure log to a file with detailed exception information.
    
    Args:
        model_name: Name of the model that failed
        attack_name: Name of the attack that failed
        exception: The exception that occurred
        results_folder: Folder to save the failure log
    """
    
    # Create failure logs directory
    failure_logs_dir = os.path.join(results_folder, "failure_logs")
    os.makedirs(failure_logs_dir, exist_ok=True)
    
    # Create filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"failure_log_{model_name}_{attack_name}_{timestamp}.txt"
    filepath = os.path.join(failure_logs_dir, filename)
    
    # Write failure log
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"FAILURE LOG\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Attack: {attack_name}\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Exception Type: {type(exception).__name__}\n")
        f.write(f"Exception Message: {str(exception)}\n")
        f.write(f"\nFull Traceback:\n")
        f.write(f"{'=' * 50}\n")
        f.write(traceback.format_exc())
        f.write(f"\n{'=' * 50}\n")
    
    print(f"    📝 Failure log saved to: {filepath}")


def run_attacks_on_models(
    model_names: List[str], 
    attack_names: List[str], 
    data_loader: DataLoader,
    device: str = 'cuda',
    results_folder: str = "",
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

    save_results = results_folder != ""
    
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
        result = load_model_imagenette(model_path, model_name, device=device)
        
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

                    # Add accuracy drop metrics
                    result_dict['clean_accuracy'] = ev.clean_accuracy
                    result_dict['accuracy_drop'] = ev.accuracy_drop
                    result_dict['relative_accuracy_drop'] = ev.relative_accuracy_drop
                    result_dict['asr_unconditional'] = ev.asr_unconditional
                    result_dict['asr_conditional'] = ev.asr_conditional


                    result_dict['time'] = ev.time
                    result_dict['n_samples'] = ev.n_samples


                    model_results.append(result_dict)
                    print(f"    ✅ {attack_name}: {ev.acc:.2f}% accuracy")
                else:
                    print(f"    ⚠️  {attack_name}: No results generated")
                    
            except Exception as e:
                print(f"    ❌ {attack_name} failed: {str(e)}")
                if save_results:
                    save_failure_log(model_name, attack_name, e, results_folder)
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

# Example usage of the comprehensive attack evaluation
if __name__ == "__main__":
    
    # Run simple example for testing
    print(f"\n🎉 Simple example completed!")
    print(f"📁 Results saved to: results/attacks/imagenette_models/")
    
    model_names = [
        ModelNames().resnet18,
        ModelNames().vgg16,
        ModelNames().densenet121,
        ModelNames().mobilenet_v2,
        ModelNames().efficientnet_b0
    ]
    
    attack_names = AttackNames().all_attack_names
    attack_names = [AttackNames().PGDRS, AttackNames().PGDRSL2, AttackNames().SPSA]
    save_folder = "data/attacks/imagenette_models"

    attack_and_save_images_multiple(model_names, attack_names, images_per_attack=100, successfully_attacked_images_folder=save_folder)
s