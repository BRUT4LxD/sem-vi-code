import numpy as np
import itertools
from architectures.efficientnet import EfficientNetB0
from architectures.mobilenetv2 import MobileNetV2
from architectures.resnet import ResNet18, ResNet50, ResNet101, ResNet152
from architectures.densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201
from architectures.sample_conv import ConvNetMNIST, ConvNetCIFAR
from architectures.vgg import VGG11, VGG13, VGG16, VGG19
from attacks.system_under_attack import multiattack, single_attack
from attacks.white_box.vanila import VANILA
from config.imagenette_models import get_imagenette_pretrained_models
from data_eng.dataset_loader import load_MNIST, load_imagenette, load_CIFAR10
from torchvision import transforms
import torch
from data_eng.io import load_model
from domain.model_config import ModelConfig
from evaluation.metrics import evaluate_attack, evaluate_model
from evaluation.validation import simple_validation
from evaluation.visualization import plot_multiattacked_images, simple_visualize
from config.model_classes import mnist_classes, imagenette_classes
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from attacks.white_box import PGD, FGSM, FFGSM, OnePixel, get_all_white_box_attack
from attacks.black_box import get_all_black_box_attack
from domain.attack_result import AttackResult
import math
from training import train_all_archs_for_imagenette

from training.efficient_net_imagenette import train_all_efficient_net
from training.mobilenet_v2_imagenette import train_all_mobilenet
from training.resnet_imagenette import train_all_resnet
from training.vgg_imagenette import train_all_vgg
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)

_, test_loader = load_imagenette(transform=transform, batch_size=1)

model_configs = get_imagenette_pretrained_models()
model_configs = [
    ModelConfig(MobileNetV2(), 'models/mobilenetv2imagenette.pt', True),
    ModelConfig(VGG11(), 'models/vgg11imagenette.pt', True),
    ModelConfig(VGG13(), 'models/vgg13imagenette.pt', True),
    ModelConfig(VGG16(), 'models/vgg16imagenette.pt', True),
    ModelConfig(VGG19(), 'models/vgg19imagenette.pt', True),
]

for config in model_configs:
    model = config.model.to(device)
    attacks = get_all_white_box_attack(model)
    multiattack_result = multiattack(
        attacks, test_loader, device, print_results=False, iterations=200, save_results=True)

    plot_multiattacked_images(
        multiattack_result, imagenette_classes, save_visualization=True, visualize=False)

    del model, multiattack_result
