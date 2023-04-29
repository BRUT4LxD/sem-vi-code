import numpy as np
import itertools
from architectures.resnet import ResNet18, ResNet50, ResNet101, ResNet152
from architectures.densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201
from architectures.sample_conv import ConvNetMNIST, ConvNetCIFAR
from attacks.system_under_attack import multiattack, single_attack
from config.imagenette_models import get_imagenette_pretrained_models
from data_eng.dataset_loader import load_MNIST, load_imagenette, load_CIFAR10
from torchvision import transforms
import torch
from data_eng.io import load_model
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

_, test_loader = load_imagenette(transform=transform, batch_size=5)

train_all_mobilenet(50)

for config in get_imagenette_pretrained_models():
    model = config.model.to(device)
    multiattack_result = multiattack(
        get_all_white_box_attack(model), test_loader, device, print_results=False)

    plot_multiattacked_images(
        multiattack_result, imagenette_classes, save_visualization=True, visualize=False)

    del model, multiattack_result
