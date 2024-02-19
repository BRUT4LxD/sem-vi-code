import torch
import torch.nn as nn
from architectures.sample_conv import ConvNetCIFAR, ConvNetImageNet, ConvNetMNIST

from data_eng.dataset_loader import DatasetLoader, DatasetType
from data_eng.io import load_model
from evaluation.validation import Validation
from training.train import Training
from config.cifar_classes import CifarClasses

class CNNSimpleModels(object):

  @staticmethod
  def get_model_by_dataset_type(datasetType: str):
    if datasetType == DatasetType.MNIST:
      return ConvNetMNIST()
    if datasetType == DatasetType.CIFAR10:
      return ConvNetCIFAR()
    if datasetType == DatasetType.IMAGENETTE:
      return ConvNetImageNet()

    raise Exception('Invalid dataset type: ' + datasetType)

class CNNSimpleTrainer:
  @staticmethod
  def train(datasetType: str, num_epochs: int, learning_rate: float, batch_size: int, model_save_path: str, load_pretrained: bool):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = DatasetLoader.get_dataset_by_type(datasetType)
    model_instance = CNNSimpleModels.get_model_by_dataset_type(datasetType)
    model = load_model(model_instance, model_save_path) if load_pretrained else model_instance.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    Training.simple_train(model, criterion, optimizer, train_loader,
                num_epochs=num_epochs,
                device=device,
                save_model_path=model_save_path)

    Validation.simple_validation(model, test_loader, batch_size, CifarClasses.get_classes(), device=device)

