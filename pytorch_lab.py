import numpy as np
import itertools
from architectures.sample_conv import ConvNetMNIST, ConvNetCIFAR
from attacks.system_under_attack import multiattack, single_attack
from data_eng.dataset_loader import load_MNIST, load_imagenette, load_CIFAR10
from torchvision import transforms
import torch
from data_eng.io import load_model
from evaluation.metrics import evaluate_attack, evaluate_model
from evaluation.visualization import plot_attacked_images, simple_visualize
from constants.model_classes import mnist_classes
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from attacks.white_box import PGD, FGSM, FFGSM, OnePixel, get_all_white_box_attack
from domain.attack_result import AttackResult
import math

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_SAVE_PATH = './models/cnn-mnist.pt'

model = load_model(ConvNetMNIST().to(device), MODEL_SAVE_PATH)

_, test_loader = load_MNIST(batch_size=100)

# multiattack_result = multiattack([PGD(model), FGSM(model), FFGSM(model),
#                                   OnePixel(model)], test_loader, device)

multiattack_result = multiattack(
    get_all_white_box_attack(model), test_loader, device)

plot_attacked_images(multiattack_result.attack_results, multiattack_result.adv_images, multiattack_result.eval_scores)
