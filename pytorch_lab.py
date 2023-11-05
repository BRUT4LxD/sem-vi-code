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

all_model_names = ModelNames().model_names
model_name = ModelNames.resnet18
model = ImageNetModels.get_model(model_name)

# measure time for attack

for model_name in all_model_names:
    model = ImageNetModels.get_model(model_name)
    model = model.to(device)
    model = model.train()
    attack = AttackFactory().get_attack(attack_name=AttackNames.DeepFool, model=model)
    attack.model = model.to(device)
    start = time.time()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    adv_images = attack(images, labels)
    end = (time.time() - start) * 1000
    print(f"({model_name}) Attack took {end} ms")
    del model, attack, images, labels, adv_images
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
