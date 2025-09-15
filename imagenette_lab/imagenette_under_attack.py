import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from attacks.attack import Attack
from attacks.system_under_attack import SystemUnderAttack
from attacks.white_box.fgsm import FGSM
from config.imagenette_classes import ImageNetteClasses
from data_eng.dataset_loader import load_imagenette
from data_eng.io import load_model_imagenette
from domain.attack.attack_result import AttackResult
from domain.model.model_names import ModelNames
from evaluation.metrics import Metrics
from shared.model_utils import ModelUtils

def attack_images_imagenette(attack: Attack, data_loader: DataLoader, save_results: bool = False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = attack.model

    model.eval()
    model.to(device)

    num_classes = len(ImageNetteClasses.get_classes())

    attack_results = []

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

    print(ev)

    if save_results:
        save_path = f'{attack.model_name}_{attack.attack}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(save_path, 'w') as file:
            file.write(str(ev))
    
    return ev


# Load your trained ResNet18 model
model_path = "./models/imagenette/resnet18_advanced.pt"
model_name = ModelNames().resnet18
result = load_model_imagenette(model_path, model_name, device='cuda')

if not result['success']:
    print(f"❌ Failed to load model: {result['error']}")
    exit()

model = result['model']
print(f"✅ Loaded model: {result['model_name']}")

# Create FGSM attack
attack = FGSM(model)

# Get test data (small subset for quick testing)
_, test_loader = load_imagenette(batch_size=1, test_subset_size=-1)

attack_images_imagenette(attack, test_loader, save_results=False)