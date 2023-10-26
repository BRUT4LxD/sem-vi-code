from data_eng.dataset_loader import load_imagenette
from torchvision import transforms
import torch
from data_eng.pretrained_model_downloader import PretrainedModelDownloader
from domain.model_names import ModelNames
from evaluation.validation import Validation
from config.model_classes import imagenette_classes
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

for model_name in all_model_names:
    if model_name == ModelNames.resnet18 or model_name == ModelNames.resnet50:
        continue

    try:
        model = ImageNetModels.get_model(model_name)
        model = SetupPretraining.setup_imagenette(model)
        # Unfreeze the last 5 layers
        for child in list(model.children())[-5:]:
            for param in child.parameters():
                param.requires_grad = True

        model = model.to(device)
        save_model_path = f'./models/transfer/{model_name}_imagenette.pt'
        Training.train_imagenette(model=model, train_loader=train_loader, device=device, learning_rate=0.0001, num_epochs=5, save_model_path=save_model_path, model_name=model_name)
        Validation.simple_validation(model=model, test_loader=test_loader, classes=imagenette_classes, device=device)
    except Exception as e:
        print(e)
        continue

# torch.cuda.empty_cache()

# for config in model_configs:
#     model = config.model.to(device)
#     attacks = get_all_attacks(model)

#     multiattack_result = multiattack(
#         attacks, test_loader, device, print_results=True, iterations=5, save_results=True)

#     plot_multiattacked_images(
#         multiattack_result, imagenette_classes, save_visualization=True, visualize=False)

#     del model, multiattack_result
