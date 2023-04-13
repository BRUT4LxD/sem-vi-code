import numpy as np
import itertools
from architectures.sample_conv import ConvNetMNIST
from attacks.pgd import PGD_MN
from utils.dataset_loader import load_MNIST, load_imagenette
from torchvision import transforms
import torch
from utils.io import load_model
from utils.metrics import evaluate_attack, evaluate_model

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_SAVE_PATH = './models/cnn-mnist.pt'

model = load_model(ConvNetMNIST().to(device), MODEL_SAVE_PATH)


_, test_loader = load_MNIST()

att = PGD_MN(model)
attack_results = att.forward(test_loader)


print(attack_results[1])

attack_eval = evaluate_attack(attack_results, 10)

print(attack_eval)
print(attack_eval.conf_matrix)
