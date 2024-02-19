import torch
import torch.nn as nn

from architectures.sample_conv import ConvNetCIFAR
from data_eng.dataset_loader import load_CIFAR10
from data_eng.io import load_model
from evaluation.validation import Validation
from training.train import Training
from config.cifar_classes import CifarClasses

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 4
num_epochs = 5
learning_rate = 0.001
MODEL_SAVE_PATH = './models/cnn-cifar.pt'
LOAD_PRETRAINED = False

train_loader, test_loader = load_CIFAR10()

model_instance = ConvNetCIFAR()

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

Validation.simple_validation(model, test_loader, batch_size, CifarClasses.get_classes(), device=device)
