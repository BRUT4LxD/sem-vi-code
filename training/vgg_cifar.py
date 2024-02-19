import torch
import torch.nn as nn
from architectures.vgg import VGG
from data_eng.dataset_loader import load_CIFAR10
from config.cifar_classes import CifarClasses
from evaluation.validation import Validation
from training.train import Training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 4
num_epochs = 5
learning_rate = 0.001
SAVE_MODEL_PATH = 'models/vgg16cifar.pt'

train_loader, test_loader = load_CIFAR10()

model = VGG(vgg_name='VGG16').to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

Training.simple_train(model=model,
             loss_fn=criterion,
             optimizer=optimizer,
             train_loader=train_loader,
             num_epochs=num_epochs,
             device=device,
             save_model_path=SAVE_MODEL_PATH)

Validation.simple_validation(model=model,
                  test_loader=test_loader,
                  batch_size=batch_size,
                  classes=CifarClasses.get_classes(),
                  device=device)
