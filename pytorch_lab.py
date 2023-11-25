from attacks import get_all_attacks
from attacks.attack_factory import AttackFactory
from attacks.attack_names import AttackNames
from attacks.simple_attacks import SimpleAttacks
from attacks.transferability import Transferability
from attacks.white_box.fgsm import FGSM
from config.imagenette_classes import ImageNetteClasses
from data_eng.dataset_loader import DatasetLoader, load_imagenette
from torchvision import transforms
import torch
from data_eng.pretrained_model_downloader import PretrainedModelDownloader
from domain.model_names import ModelNames
from evaluation.validation import Validation
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from datetime import datetime

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

model_name = all_model_names[0]
attack_name = all_attack_names[0]

# train, test = DatasetLoader.get_attacked_imagenette_dataset(model_name=model_name, attack_name=attack_name, transform=transform)

# for images, labels in train:
#     print(labels.item())

trans_attacks = []

for attack_name in all_attack_names:
    if attack_name != AttackNames().DeepFool and attack_name != AttackNames().SparseFool and attack_name != AttackNames().FAB and attack_name != AttackNames().SPSA:
        trans_attacks.append(attack_name)

save_folder = './results/transferability4'
for model_name in [ModelNames.vgg13, ModelNames.vgg16, ModelNames.vgg19]:
    print(f'Model: {model_name:10s} Time: {datetime.today().strftime("%H:%M:%S")}')
    Transferability.transferability_attack_to_model(
        attacked_model_name=model_name,
        trans_models_names=all_model_names,
        attack_names=trans_attacks,
        save_path_folder=save_folder,
        device=device)

save_folder_trans_m2m = './results/m2mtransferability/'

attks = []
for attack_name in all_attack_names:
        attks.append(attack_name)

# Transferability.transferability_model_to_model(all_model_names, attks, save_path_folder=save_folder_trans_m2m, device=device)
# torch.cuda.empty_cache()

# for config in model_configs:
#     model = config.model.to(device)
#     attacks = get_all_attacks(model)

#     multiattack_result = multiattack(
#         attacks, test_loader, device, print_results=True, iterations=5, save_results=True)

#     plot_multiattacked_images(
#         multiattack_result, imagenette_classes, save_visualization=True, visualize=False)

#     del model, multiattack_result