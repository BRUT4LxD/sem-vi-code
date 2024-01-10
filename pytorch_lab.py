from attacks import get_all_attacks
from attacks.attack_factory import AttackFactory
from attacks.attack_names import AttackNames
from attacks.simple_attacks import SimpleAttacks
from attacks.transferability import Transferability
from attacks.white_box.fgsm import FGSM
from config.imagenette_classes import ImageNetteClasses
from data_eng.attacked_dataset_generator import AttackedDatasetGenerator
from data_eng.dataset_loader import DatasetLoader, DatasetType, load_imagenette
from torchvision import transforms
import torch
from data_eng.io import load_model
from data_eng.pretrained_model_downloader import PretrainedModelDownloader
from domain.model_names import ModelNames
from evaluation.adversarial_validaton import AdversarialValidation
from evaluation.validation import Validation
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from datetime import datetime
import os
import csv

from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
import torchvision.transforms as transforms 
from training.adversarial_training import AdversarialTraining
from training.train import Training
from config.imagenet_models import ImageNetModels
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from training.transfer.setup_pretraining import SetupPretraining

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, test_loader = load_imagenette(batch_size=128, shuffle=True)

all_model_names = ModelNames().all_model_names
all_attack_names = AttackNames().all_attack_names
invalid_attack_names = [AttackNames().DeepFool, AttackNames().SparseFool, AttackNames().FAB, AttackNames().SPSA, AttackNames().JSMA]
valid_attack_names = [attack_name for attack_name in all_attack_names if attack_name not in invalid_attack_names]

model_name = ModelNames().resnet18
resnet = ImageNetModels().get_model(model_name=model_name)
resnet.to(device)

iterations = 20
i = 0

# resnet = load_model(resnet, f"./models/adversarial_models/{model_name}/it{i - 1}.pt")
valid_attack_names = [att for att in valid_attack_names if att != AttackNames().PGDRSL2 and att != AttackNames().EADEN and att != AttackNames().EADL1 and att != AttackNames().Square]

optimizer = torch.optim.Adam(resnet.parameters(), lr=0.00001)
adv_save_path = f"./results/adversarial_training/{model_name}/adv_val3.txt"
save_path = f"./results/adversarial_training/{model_name}/val3.txt"
save_model_path = f"./models/adversarial_models/{model_name}/it{i}.pt"

adv = AdversarialTraining(attack_names=valid_attack_names, model=resnet, optimizer=optimizer, device=device, model_name=model_name)
while i < iterations:
  save_model_path = f"./models/adversarial_models/{model_name}/it{i}.pt"
  val_result, adv_val_result = adv.train(num_epochs=20, images_per_attack=10, attack_ratio=1, save_model_path=save_path, it_num=i)
  print(val_result.accuracy)
  print(adv_val_result.accuracy)
  i += 1



# while i < iterations:
#   adv_save_path = f"./results/adversarial_training/{model_name}/adv_val.txt"
#   save_path = f"./results/adversarial_training/{model_name}/val.txt"
#   save_model_path = f"./models/adversarial_models/{model_name}/it{i}.pt"
#   optimizer = torch.optim.Adam(resnet.parameters(), lr=0.0001)
#   attacked_test_loader = AttackedDatasetGenerator.get_attacked_imagenette_dataset_half(model=resnet, model_name=model_name, attack_names=valid_attack_names, test_subset_size=1000)
#   adv = AdversarialTraining(attack_names=valid_attack_names, model=resnet, optimizer=optimizer, device=device, model_name=model_name)
#   adv.train(num_epochs=10, images_per_attack=40, attack_ratio=0.5, save_model_path=save_model_path)
#   val_result = Validation.validate_imagenet_with_imagenette_classes(model=resnet, model_name=model_name, test_loader=test_loader, device=device)
#   val_result.append_to_csv_file(it=i, save_path=save_path)
#   adv_val_result = Validation.validate_imagenet_with_imagenette_classes(model=resnet, model_name=model_name, test_loader=attacked_test_loader, device=device)
#   adv_val_result.append_to_csv_file(it=i, save_path=adv_save_path)
#   i += 1

# Validation.validate_imagenet_with_imagenette_classes(model=resnet, model_name=model_name, test_loader=test_loader, device=device)