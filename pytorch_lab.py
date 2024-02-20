from attacks import get_all_attacks
from attacks.attack_factory import AttackFactory
from attacks.attack_names import AttackNames
from attacks.simple_attacks import SimpleAttacks
from attacks.transferability import Transferability
from attacks.white_box.fgsm import FGSM
from config.binary_models import BinaryModels
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

from domain.multiattack_result import MultiattackResult, AttackResult

from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
import torchvision.transforms as transforms
from evaluation.visualization import plot_multiattacked_images 
from training.adversarial_training import AdversarialTraining
from training.train import Training
from config.imagenet_models import ImageNetModels
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from training.transfer.setup_pretraining import SetupPretraining
from torch.utils.tensorboard import SummaryWriter

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, test_loader = load_imagenette(batch_size=1, shuffle=True)

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
lr = 0.0001
attack_ratio = 1
images_per_attack = 10
progressive_learning = True

model_name = ModelNames().resnet18
raw_resnet_model = ImageNetModels().get_model(model_name)
trained_resnet = load_model(raw_resnet_model, f"./models/imagenette/{model_name}.pt")
adv_resnet = load_model(raw_resnet_model, f"./models/adversarial_models/{model_name}/30-01-2024_01-11.pt")

model_name = ModelNames().densenet121
raw_densenet_model = ImageNetModels().get_model(model_name)
trained_densenet = load_model(raw_densenet_model, f"./models/imagenette/{model_name}.pt")
adv_densenet = load_model(raw_densenet_model, f"./models/adversarial_models/{model_name}/11-02-2024_19-54.pt")

model_name = ModelNames().mobilenet_v2
raw_mobilenet_model = ImageNetModels().get_model(model_name)
trained_mobilenet = load_model(raw_mobilenet_model, f"./models/imagenette/{model_name}.pt")
adv_mobilenet = load_model(raw_mobilenet_model, f"./models/adversarial_models/{model_name}/13-02-2024_10-49.pt")

model_name = ModelNames().efficientnet_b0
raw_efficientnet_model = ImageNetModels().get_model(model_name)
trained_efficientnet = load_model(raw_efficientnet_model, f"./models/imagenette/{model_name}.pt")
adv_efficientnet = load_model(raw_efficientnet_model, f"./models/adversarial_models/{model_name}/13-02-2024_22-59.pt")

model_name = ModelNames().vgg16
raw_vgg_model = ImageNetModels().get_model(model_name)
trained_vgg = load_model(raw_vgg_model, f"./models/imagenette/{model_name}.pt")
adv_vgg = load_model(raw_vgg_model, f"./models/adversarial_models/{model_name}/15-02-2024_07-37.pt")


model_names = [ModelNames().resnet18, ModelNames().densenet121, ModelNames().mobilenet_v2, ModelNames().efficientnet_b0, ModelNames().vgg16]
trained_models = [trained_resnet, trained_densenet, trained_mobilenet, trained_efficientnet, trained_vgg]
adv_models = [adv_resnet, adv_densenet, adv_mobilenet, adv_efficientnet, adv_vgg]


# for model_name in model_names:
#   model = ImageNetModels().get_model(model_name=model_name)
#   Training.train_imagenette(
#     model=model,
#     model_name=model_name,
#     train_loader=train_loader,
#     test_loader=test_loader,
#     num_epochs=100,
#     learning_rate=lr
#   )

#   save_model_path = f"./models/imagenette/{model_name}.pt"
#   save_model(model, save_model_path)

#   results_path = f"./results/imagenette_trained_models/{model_name}_lr{lr}.csv"
#   res = Validation.validate_imagenet_with_imagenette_classes(
#     model=model,
#     test_loader=test_loader,
#     model_name=model_name,
#     device=device
#   )

#   print(f"Validation accuracy: {res}")
#   res.save_csv(0, results_path)

# _ = adv_vgg.eval()
# _ = adv_vgg.to(device)
# Validation.validate_imagenet_with_imagenette_classes(
#   model=adv_vgg,
#   test_loader=test_loader,
#   model_name=ModelNames().vgg16,
# )

multiattacks = []
attacks_save_folder_path = f"./results/attacks/adv_models"
attack_images_save_folder_path = f"visualization/adv_models"
for adv_model in adv_models:
  _ = adv_model.to(device)
  attack_list = []
  for attack_name in valid_attack_names:
    attack_list.append(AttackFactory.get_attack(attack_name, adv_model))

  multiattack_result = SimpleAttacks.multiattack(
    attacks=attack_list,
    device=device,
    data_loader=test_loader,
    iterations=100,
    is_imagenette_model=True,
    save_folder_path=attacks_save_folder_path
  )

  # leave only successfully attacked images
  multiattack_result.attack_results[0] = [att for att in multiattack_result.attack_results[0] if att.actual != att.predicted]
  plot_multiattacked_images(
    classes_names=ImageNetteClasses.get_classes(),
    multiattack_results=multiattack_result,
    save_path_folder=attack_images_save_folder_path,
    visualize=False
  )


# diff = torch.abs(adv_image - src_image)
# diff_mask = torch.where(diff == 0, torch.tensor(1.0), torch.tensor(0.0))
# diff_mask_summed = torch.clamp(diff_mask.sum(0), 0, 1)
# diff_mask_summed = diff_mask_summed.cpu().numpy()

# plot_multiattacked_images(
#   classes_names=ImageNetteClasses.get_classes(),
#   multiattack_results=multiattacks[0]
# )

# attacked_dataloder = AttackedDatasetGenerator.get_attacked_imagenette_dataset_multimodel(
#   model_names=model_names,
#   attack_names=valid_attack_names,
#   num_of_images_per_attack=30,
#   use_test_set=True,
#   batch_size=1
# )

# for model, model_name in zip(trained_models, model_names):
#   _ = model.eval()
#   _ = model.to(device)

#   res = Validation.validate_imagenet_with_imagenette_classes(
#     model=model,
#     test_loader=attacked_dataloder,
#     model_name=model_name,
#     device=device
#   )

#   res.save_csv(0, f"./results/models_accuracy/imagenette/{model_name}/adv_val.txt")

# model = BinaryModels.resnet18()
# model_name = ModelNames().resnet18

# attacked_binary_train = [] 
# attacked_binary_test = []

# for images, labels in train_loader:
#   attacked_binary_test.append((images[0], 0))
#   attacked_binary_train.append((images[0], 1))

# attacked_binary_test = DataLoader(attacked_binary_test, batch_size=1, shuffle=True)
# attacked_binary_train = DataLoader(attacked_binary_train, batch_size=1, shuffle=True)

# Training.train_binary(
#   model=model,
#   model_name=model_name,
#   train_loader=attacked_binary_train,
#   test_loader=attacked_binary_test,
#   num_epochs=20,
#   learning_rate=lr
# )

# attacked_binary_test = AttackedDatasetGenerator.get_attacked_imagenette_dataset_multimodel_for_binary(
#   model_names=model_names,
#   attack_names=valid_attack_names,
#   num_of_images_per_attack=10,
#   use_test_set=True,
#   batch_size=1
# )


# date = datetime.now().strftime("%d-%m-%Y_%H-%M")
# writer = SummaryWriter(log_dir=f'runs/binary_training/{model_name}/{date}_lr={lr}')

# for i in range(5):
#   attacked_binary_train = AttackedDatasetGenerator.get_attacked_imagenette_dataset_multimodel_for_binary(
#     model_names=model_names,
#     attack_names=valid_attack_names,
#     num_of_images_per_attack=30,
#     use_test_set=False,
#     batch_size=1
#   )

#   Training.train_binary(
#     model=model,
#     model_name=model_name,
#     train_loader=attacked_binary_train,
#     test_loader=attacked_binary_test,
#     num_epochs=50,
#     learning_rate=lr,
#     writer=writer,
#     save_model_path=f"./models/binary/{model_name}.pt"
#   )

#   del attacked_binary_train


# save_model_path = f"./models/binary/{model_name}.pt"
# save_model(model, save_model_path)


# attacked_binary_test = AttackedDatasetGenerator.get_attacked_imagenette_dataset_multimodel_for_binary(
#   model_names=model_names,
#   attack_names=valid_attack_names,
#   num_of_images_per_attack=5,
#   use_test_set=True,
#   batch_size=1
# )

# res = Validation.validate_binary_classification(
#   model=model,
#   test_loader=attacked_binary_test,
#   model_name=model_name,
#   device=device
# )

# results_path = f"./results/imagenette_trained_models/{model_name}_lr{lr}.csv"
# print(f"Validation accuracy: {res}")
# res.save_csv(0, results_path)
