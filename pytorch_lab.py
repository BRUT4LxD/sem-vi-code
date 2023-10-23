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

all_models = ImageNetModels.get_all_resnet_models()

for model in all_models:
    model = SetupPretraining.setup_imagenette(model)
    model = model.to(device)
    Training.train_imagenette(model=model, train_loader=train_loader, device=device, learning_rate=0.001, num_epochs=20, save_model_path='./models/transfer/mobilenet_v2_imagenette.pt')
    Validation.simple_validation(model=model, test_loader=test_loader, classes=imagenette_classes, device=device)

# torch.cuda.empty_cache()

# for config in model_configs:
#     model = config.model.to(device)
#     attacks = get_all_attacks(model)

#     multiattack_result = multiattack(
#         attacks, test_loader, device, print_results=True, iterations=5, save_results=True)

#     plot_multiattacked_images(
#         multiattack_result, imagenette_classes, save_visualization=True, visualize=False)

#     del model, multiattack_result
