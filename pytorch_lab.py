import numpy as np
import itertools
from architectures.sample_conv import ConvNetMNIST
from attacks.fgsm import FGSM
from utils.dataset_loader import load_MNIST, load_imagenette
from torchvision import transforms
import torch
from utils.io import load_model
from utils.metrics import evaluate_attack

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_SAVE_PATH = './models/cnn-mnist.pt'
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ]
)


model = load_model(ConvNetMNIST().to(device), MODEL_SAVE_PATH)

print(model)
_, test_dataset = load_MNIST(transform)

test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False)

fgsm = FGSM(model)
attack_results = fgsm.attack(model, test_loader)

attack_eval = evaluate_attack(attack_results, 10)

print(attack_eval)
print(attack_eval.conf_matrix)
