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
from data_eng.io import load_model, save_model
from data_eng.pretrained_model_downloader import PretrainedModelDownloader
from domain.attack_result import AttackedImageResult
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
from torch.utils.tensorboard import SummaryWriter

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, test_loader = load_imagenette(batch_size=16, shuffle=True)

all_model_names = ModelNames().all_model_names
all_attack_names = AttackNames().all_attack_names
invalid_attack_names = [AttackNames().DeepFool, AttackNames().SparseFool, AttackNames().FAB, AttackNames().SPSA, AttackNames().JSMA]
valid_attack_names = [attack_name for attack_name in all_attack_names if attack_name not in invalid_attack_names]

model_name = ModelNames().resnet18
resnet = ImageNetModels().get_model(model_name=model_name)

# resnet = load_model(resnet, f"./models/adversarial_models/{model_name}/22-01-2024_00-33.pt")
resnet.to(device)

iterations = 50
i = 0

# resnet = load_model(resnet, f"./models/adversarial_models/{model_name}/it{i - 1}.pt")
valid_attack_names = [att for att in valid_attack_names if att != AttackNames().PGDRSL2 and att != AttackNames().EADEN and att != AttackNames().EADL1 and att != AttackNames().Square]
lr = 0.00001
attack_ratio = 1
images_per_attack = 10
progressive_learning = True

model_names = [ModelNames().vgg16, ModelNames().resnet18]

# adv_save_path = f"./results/adversarial_training/{model_name}/adv_val.txt"
# save_path = f"./results/adversarial_training/{model_name}/val.txt"


# for model_name in model_names:
#   current_model = ImageNetModels().get_model(model_name=model_name)
#   adv = AdversarialTraining(attack_names=valid_attack_names, model=current_model, learning_rate=lr, device=device, model_name=model_name)

#   date = datetime.now().strftime("%d-%m-%Y_%H-%M")
#   save_model_path = f"./models/adversarial_models/{model_name}/{date}.pt"
#   writer = SummaryWriter(log_dir=f'runs/adversarial_training/{model_name}/{date}_lr={lr}_ar={attack_ratio}_imgs={images_per_attack*len(valid_attack_names)}_prgrsv={progressive_learning}')
#   adv.train_progressive(
#     epochs_per_iter=30,
#     writer=writer,
#     iterations=iterations,
#     images_per_attack=images_per_attack,
#     attack_ratio=1,
#     save_model_path=save_model_path)
#   writer.close()

for model_name in model_names:
  model = ImageNetModels().get_model(model_name=model_name)
  Training.train_imagenette(
    model=model,
    model_name=model_name,
    train_loader=train_loader,
    test_loader=test_loader,
    num_epochs=100,
    learning_rate=lr
  )

  save_model_path = f"./models/imagenette/{model_name}.pt"
  save_model(model, save_model_path)

  results_path = f"./results/imagenette_trained_models/{model_name}_lr{lr}.csv"
  res = Validation.validate_imagenet_with_imagenette_classes(
    model=model,
    test_loader=test_loader,
    model_name=model_name,
    device=device
  )

  print(f"Validation accuracy: {res}")
  res.save_csv(0, results_path)


for model_name in model_names:
  model = ImageNetModels().get_model(model_name=model_name)
  _ = model.eval()
  _ = model.to(device)

  model_name = model_names[i]
  adv_res = Validation.validate_imagenet_with_imagenette_classes(
    model=model,
    test_loader=test_loader,
    model_name=model_name,
    device=device
  )

  adv_res.save_csv(0, f"./results/adversarial_training/{model_name}/adv_val_{len(attacked_dataloader.dataset)}_untrained.txt")

  res = Validation.validate_imagenet_with_imagenette_classes(
    model=model,
    test_loader=test_loader,
    model_name=model_name,
    device=device
  )

  res.save_csv(0, f"./results/adversarial_training/{model_name}/val_{len(test_loader.dataset)}_untrained.txt")

  i = i + 1