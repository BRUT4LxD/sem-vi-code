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

model_names = [ModelNames().resnet18, ModelNames().densenet121, ModelNames().mobilenet_v2, ModelNames().efficientnet_b0, ModelNames().vgg16]

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



adv_models = []

model_name = ModelNames().resnet18
raw_model = ImageNetModels().get_model(model_name=model_name)
adv_resnet = load_model(raw_model, f"./models/adversarial_models/{model_name}/04-02-2024_00-47.pt")


model_name = ModelNames().densenet121
raw_model = ImageNetModels().get_model(model_name=model_name)
adv_densenet = load_model(raw_model, f"./models/adversarial_models/{model_name}/11-02-2024_19-54.pt")

model_name = ModelNames().mobilenet_v2
raw_model = ImageNetModels().get_model(model_name=model_name)
adv_mobilenet = load_model(raw_model, f"./models/adversarial_models/{model_name}/13-02-2024_10-49.pt")

model_name = ModelNames().efficientnet_b0
raw_model = ImageNetModels().get_model(model_name=model_name)
adv_efficientnet = load_model(raw_model, f"./models/adversarial_models/{model_name}/13-02-2024_22-59.pt")

model_name = ModelNames().vgg16
raw_model = ImageNetModels().get_model(model_name=model_name)
adv_vgg = load_model(raw_model, f"./models/adversarial_models/{model_name}/15-02-2024_07-37.pt")

adv_models.append(adv_resnet)
adv_models.append(adv_densenet)
adv_models.append(adv_mobilenet)
adv_models.append(adv_efficientnet)
adv_models.append(adv_vgg)

attacked_dataset: list['AttackedImageResult'] = []

# generate adversarial examples for all model_names and valid_attacks
for model_name in tqdm(model_names):
  current_model = ImageNetModels().get_model(model_name=model_name)
  current_model = current_model.to(device)

  for attack_name in tqdm(valid_attack_names):
    attack = AttackFactory.get_attack(attack_name=attack_name, model=current_model)
    attacked_subset = SimpleAttacks.get_attacked_imagenette_images(
      model=current_model,
      attack_name=attack_name,
      model_name=model_name,
      batch_size=1,
      num_of_images=30,
      use_test_set=True
    )

    attacked_dataset.extend(attacked_subset)


att_dataset = [(att.adv_image, att.label) for att in attacked_dataset]
attacked_dataloader = DataLoader(att_dataset, batch_size=16, shuffle=True)

res = Validation.validate_imagenet_with_imagenette_classes(
  model=adv_vgg,
  test_loader=test_loader,
  model_name=ModelNames().vgg16,
  device=device
)

res.save_csv(0, f"./results/adversarial_training/{ModelNames().vgg16}/val_{len(test_loader.dataset)}.txt")

i = 0
for adv_model in adv_models:
  _ = adv_model.eval()
  _ = adv_model.to(device)

  model_name = model_names[i]
  adv_res = Validation.validate_imagenet_with_imagenette_classes(
    model=adv_model,
    test_loader=attacked_dataloader,
    model_name=model_name,
    device=device
  )

  adv_res.save_csv(0, f"./results/adversarial_training/{model_name}/adv_val_{len(attacked_dataloader.dataset)}.txt")

  res = Validation.validate_imagenet_with_imagenette_classes(
    model=adv_model,
    test_loader=test_loader,
    model_name=model_name,
    device=device
  )

  res.save_csv(0, f"./results/adversarial_training/{model_name}/val_{len(test_loader.dataset)}.txt")

  i = i + 1

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