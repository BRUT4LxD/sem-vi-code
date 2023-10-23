from data_eng.dataset_loader import load_imagenette
from torchvision import transforms
import torch
from data_eng.io import load_model
from data_eng.pretrained_model_downloader import PretrainedModelDownloader
from domain.model_names import ModelNames
from evaluation.validation import Validation
from config.model_classes import mnist_classes, imagenette_classes
from training import train_all_archs_for_imagenette
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
from training.train import Training

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

# model_configs = get_imagenette_pretrained_models()

model = PretrainedModelDownloader.download_model(ModelNames.mobilenet_v2)
model = SetupPretraining.setup_imagenette(model)
model = model.to(device)

print(model.training)

Training.train_imagenette(model=model, train_loader=train_loader, device=device, learning_rate=0.001, num_epochs=20, save_model_path='./models/transfer/mobilenet_v2_imagenette.pt')

Validation.simple_validation(model=model, test_loader=test_loader, classes=imagenette_classes, device=device)

resnet = PretrainedModelDownloader.download_model(ModelNames.resnet152)

resnet = resnet.to(device)
resnet.eval()
for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)
    outputs = resnet(images)
    _, predictions = torch.max(outputs, 1)
    print(predictions)

    # plt.figure()
    # plt.imshow(images[0].cpu().permute(1, 2, 0))
    # plt.title(labels[0].item())
    # plt.show()

# torch.cuda.empty_cache()

# for config in model_configs:
#     model = config.model.to(device)
#     attacks = get_all_attacks(model)

#     multiattack_result = multiattack(
#         attacks, test_loader, device, print_results=True, iterations=5, save_results=True)

#     plot_multiattacked_images(
#         multiattack_result, imagenette_classes, save_visualization=True, visualize=False)

#     del model, multiattack_result
