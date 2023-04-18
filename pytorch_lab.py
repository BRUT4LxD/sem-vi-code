import numpy as np
import itertools
from architectures.sample_conv import ConvNetMNIST, ConvNetCIFAR
from data_eng.dataset_loader import load_MNIST, load_imagenette, load_CIFAR10
from torchvision import transforms
import torch
from data_eng.io import load_model
from evaluation.metrics import evaluate_attack, evaluate_model
from evaluation.visualization import simple_visualize
from constants.model_classes import mnist_classes
from attacks.pgd_mn import PGD_MN
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from attacks.white_box import PGD
from domain.attack_result import AttackResult
import math

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_SAVE_PATH = './models/cnn-mnist.pt'

model = load_model(ConvNetMNIST().to(device), MODEL_SAVE_PATH)

transform = transforms.Compose([
    transforms.ToTensor()
])

_, test_loader = load_MNIST(transform=transform, batch_size=1000)

atk = PGD(model, eps=8/255)
for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)
    start = datetime.datetime.now()
    adv_images = atk(images, labels)
    attack_res = AttackResult.create_from_adv_image(model, adv_images, labels)
    end = datetime.datetime.now()
    ev = evaluate_attack(attack_res, 10)
    print('Samples: {}, Metrics: {} ({} ms)'.format(labels.shape[0], ev,
          int((end-start).total_seconds()*1000)))