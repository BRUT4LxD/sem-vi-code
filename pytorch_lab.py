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
model = ImageNetModels.get_model(ModelNames.resnet101)
model = model.to(device)
# Validation.validate_imagenet_with_imagenette_classes(model=model, test_loader=test_loader, device=device)
attack_images(FGSM(model), ModelNames.resnet101, images_to_attack=10)

# torch.cuda.empty_cache()

# for config in model_configs:
#     model = config.model.to(device)
#     attacks = get_all_attacks(model)

#     multiattack_result = multiattack(
#         attacks, test_loader, device, print_results=True, iterations=5, save_results=True)

#     plot_multiattacked_images(
#         multiattack_result, imagenette_classes, save_visualization=True, visualize=False)

#     del model, multiattack_result
