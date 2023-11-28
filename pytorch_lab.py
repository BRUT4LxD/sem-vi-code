from attacks import get_all_attacks
from attacks.attack_factory import AttackFactory
from attacks.attack_names import AttackNames
from attacks.simple_attacks import SimpleAttacks
from attacks.transferability import Transferability
from attacks.white_box.fgsm import FGSM
from config.imagenette_classes import ImageNetteClasses
from data_eng.dataset_loader import DatasetLoader, DatasetType, load_imagenette
from torchvision import transforms
import torch
from data_eng.pretrained_model_downloader import PretrainedModelDownloader
from domain.model_names import ModelNames
from evaluation.validation import Validation
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from datetime import datetime

from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
from training.train import Training
from config.imagenet_models import ImageNetModels

from training.transfer.setup_pretraining import SetupPretraining

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

all_model_names = ModelNames().all_model_names
all_attack_names = AttackNames().all_attack_names
invalid_attack_names = [AttackNames().DeepFool, AttackNames().SparseFool, AttackNames().FAB, AttackNames().SPSA, AttackNames().JSMA]
valid_attack_names = [attack_name for attack_name in all_attack_names if attack_name not in invalid_attack_names]
headers = ["Attacks"] + all_model_names
print(headers)


save_folder_path = './results/transferability_100'
Transferability.transferability_attack2(
  model_names=all_model_names,
  attack_names=valid_attack_names,
  images_per_attack=100,
  attacking_batch_size=16,
  model_batch_size=2,
  save_folder_path=save_folder_path,
  print_results=True,
  device=device)

