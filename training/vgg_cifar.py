import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from architectures.vgg import VGG
from utils.dataset_loader import load_CIFAR10
from utils.io import save_model
from constants.model_classes import cifar_classes
from utils.train import simple_train
from utils.validation import simple_validation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 4
num_epochs = 5
learning_rate = 0.001
SAVE_MODEL_PATH = 'models/vgg16cifar.pt'

train_loader, test_loader = load_CIFAR10()

model = VGG(vgg_name='VGG16').to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

simple_train(model=model,
             loss_fn=criterion,
             optimizer=optimizer,
             train_loader=train_loader,
             num_epochs=num_epochs,
             device=device,
             SAVE_MODEL_PATH=SAVE_MODEL_PATH)

simple_validation(model=model,
                  test_loader=test_loader,
                  batch_size=batch_size,
                  classes=cifar_classes,
                  device=device)
