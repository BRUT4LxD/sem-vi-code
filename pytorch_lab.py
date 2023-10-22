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
from evaluation.metrics import evaluate_attack, evaluate_model
from evaluation.validation import simple_validation
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

PretrainedModelDownloader

_, test_loader = load_imagenette(transform=transform, batch_size=1)

model_configs = get_imagenette_pretrained_models()

# transfer_train_efficient_net(num_epochs=4)

for model_config in model_configs:
    model = model_config.model
    model = model.to(device)
    print(model.__class__.__name__)
    simple_validation(model, test_loader, imagenette_classes, device=device)
# attack_images(FGSM(model), DatasetType.IMAGENETTE, images_to_attack=1)

torch.cuda.empty_cache()

# for config in model_configs:
#     model = config.model.to(device)
#     attacks = get_all_attacks(model)

#     multiattack_result = multiattack(
#         attacks, test_loader, device, print_results=True, iterations=5, save_results=True)

#     plot_multiattacked_images(
#         multiattack_result, imagenette_classes, save_visualization=True, visualize=False)

#     del model, multiattack_result

# model = torch.hub.load('ultralytics/yolov5',
#                        YOLOv5Models.NANO.value, pretrained=True)

# model = model.to(device)

# multiattack_result = None
# attack = FGSM(model)
# att_res = None
# i = 0
# for images, labels in test_loader:
#     if i == 1:
#         break
#     i += 1
#     images, labels = images.to(device), labels.to(device)
#     res = model(images)
#     print(f'res shape: {res.shape}')
#     print(f'res shape1: {res[0]}')
#     print(f'res shape2: {res[0,0,:]}')
# adv_images = attack(images, labels)

# att_res = AttackResult.create_from_adv_image(
#     attack.model, adv_images, images, labels, attack.model_name, attack.attack)
# print(f'res: {res}')


# plot_multiattacked_images(
#     multiattack_result, imagenette_classes, save_visualization=True, visualize=True)


# flat = res.view(-1, 85)
# print(flat[0, :])
# preds = flat[:, 5:]
# print(f'preds: {preds.shape}')

# preds = F.softmax(preds, dim=1)
# print(f'preds: {preds.shape}')


# for i, p in enumerate(preds):
#     if i > 0:
#         break

#     sort_res2 = p.topk(k=5)
#     if sort_res2.values[0] < 0.5:
#         continue
#     print(f'sort_res: {sort_res2.indices}')
#     print(f'sort_res: {sort_res2.values > 0.5}')

# # take top 5 preds
# xx = preds.argsort(descending=True, dim=1)

# print(xx[0, :])
# print(xx[0, :5])
# print(f'xx: {xx.shape}')
# print(f'preds: {preds.shape}')
