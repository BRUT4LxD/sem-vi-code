from attacks import get_all_attacks
from attacks.attack_factory import AttackFactory
from attacks.attack_names import AttackNames
from attacks.system_under_attack import attack_images
from attacks.white_box.fgsm import FGSM
from config.imagenette_classes import ImageNetteClasses
from data_eng.dataset_loader import load_imagenette
from torchvision import transforms
import torch
from data_eng.pretrained_model_downloader import PretrainedModelDownloader
from domain.model_names import ModelNames
from evaluation.validation import Validation
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
from training.train import Training
from config.imagenet_models import ImageNetModels

from training.transfer.setup_pretraining import SetupPretraining

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)

train_loader, test_loader = load_imagenette(transform=transform, batch_size=1)

all_model_names = ModelNames().all_model_names
all_attack_names = AttackNames().all_attack_names

# measure time for attack

for attack_name in tqdm(all_attack_names):
    if attack_name == AttackNames().DeepFool:
        continue

    if attack_name == AttackNames().FAB:
        continue

    if attack_name == AttackNames().SparseFool:
        continue

    if attack_name == AttackNames().JSMA:
        continue

    for model_name in all_model_names:
        model = ImageNetModels.get_model(model_name)
        model = model.to(device)
        attack = AttackFactory().get_attack(attack_name=attack_name, model=model)
        try:
            attack_images(
                attack=attack,
                model_name=model_name,
                data_loader=train_loader,
                images_to_attack=20,
                save_results=True,
                save_base_path="./data/attacked_imagenette_train")
        except Exception as e:
            print(f"Error: {e}")
            continue
        torch.cuda.empty_cache()



# torch.cuda.empty_cache()

# for config in model_configs:
#     model = config.model.to(device)
#     attacks = get_all_attacks(model)

#     multiattack_result = multiattack(
#         attacks, test_loader, device, print_results=True, iterations=5, save_results=True)

#     plot_multiattacked_images(
#         multiattack_result, imagenette_classes, save_visualization=True, visualize=False)

#     del model, multiattack_result
