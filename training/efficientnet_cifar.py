import torch
import torch.nn as nn
from architectures.efficientnet import EfficientNetB0
from data_eng.dataset_loader import load_CIFAR10
from config.cifar_classes import CifarClasses
from data_eng.io import load_model
from evaluation.validation import Validation
from training.train import Training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 64
num_epochs = 5
learning_rate = 0.001
MODEL_SAVE_PATH = 'models/efficientnetb0cifar.pt'
LOAD_PRETRAINED = False

train_loader, test_loader = load_CIFAR10()

model_instance = EfficientNetB0()

model = load_model(model_instance,
                   MODEL_SAVE_PATH) if LOAD_PRETRAINED else model_instance.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

Training.simple_train(model=model,
             loss_fn=criterion,
             optimizer=optimizer,
             train_loader=train_loader,
             num_epochs=num_epochs,
             device=device,
             save_model_path=MODEL_SAVE_PATH)

Validation.simple_validation(model=model,
                  test_loader=test_loader,
                  batch_size=batch_size,
                  classes=CifarClasses.get_classes(),
                  device=device)
