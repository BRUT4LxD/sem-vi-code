import time
import numpy as np
import torch.nn.functional as F
import itertools
from architectures.efficientnet import EfficientNetB0
from architectures.mobilenetv2 import MobileNetV2
from architectures.resnet import ResNet18, ResNet50, ResNet101, ResNet152
from architectures.densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201
from architectures.sample_conv import ConvNetMNIST, ConvNetCIFAR
from architectures.vgg import VGG11, VGG13, VGG16, VGG19
from attacks import get_all_attacks
from attacks.system_under_attack import attack_images, multiattack, single_attack, transferability_attack
from attacks.white_box.pgdrs import PGDRS
from attacks.white_box.pgdrsl2 import PGDRSL2
from attacks.white_box.rfgsm import RFGSM
from attacks.white_box.sinifgsm import SINIFGSM
from attacks.white_box.spsa import SPSA
from attacks.white_box.tifgsm import TIFGSM
from attacks.white_box.tpgd import TPGD
from attacks.white_box.upgd import UPGD
from attacks.white_box.vmifgsm import VMIFGSM
from config.imagenette_models import get_imagenette_pretrained_models
from config.yolo_models import YOLOv5Models
from data_eng.dataset_loader import DatasetType, load_MNIST, load_imagenette, load_CIFAR10
from torchvision import transforms
import torch
from data_eng.io import load_model
from data_eng.pretrained_model_downloader import PretrainedModelDownloader
from domain.model_config import ModelConfig
from domain.model_names import ModelNames
from evaluation.metrics import evaluate_attack, evaluate_model
from evaluation.validation import Validation
from evaluation.visualization import plot_multiattacked_images, simple_visualize
from config.model_classes import mnist_classes, imagenette_classes
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from attacks.white_box import PGD, FGSM, FFGSM, OnePixel, get_all_white_box_attack, APGD
from attacks.black_box import get_all_black_box_attack
from domain.attack_result import AttackResult
import math
from training import train_all_archs_for_imagenette

from training.efficient_net_imagenette import train_all_efficient_net, transfer_train_efficient_net
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

_, test_loader = load_imagenette(transform=transform, batch_size=1, test_subset_size=500, shuffle=True)

# model_configs = get_imagenette_pretrained_models()


Validation.validate_model_by_name(ModelNames.resnet50, test_loader, imagenette_classes, device=device)

torch.cuda.empty_cache()

# for config in model_configs:
#     model = config.model.to(device)
#     attacks = get_all_attacks(model)

#     multiattack_result = multiattack(
#         attacks, test_loader, device, print_results=True, iterations=5, save_results=True)

#     plot_multiattacked_images(
#         multiattack_result, imagenette_classes, save_visualization=True, visualize=False)

#     del model, multiattack_result
