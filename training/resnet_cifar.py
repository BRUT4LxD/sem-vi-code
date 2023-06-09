import torch
import torch.nn as nn
from architectures.resnet import ResNet50
from data_eng.dataset_loader import load_CIFAR10
from config.model_classes import cifar_classes
from data_eng.io import load_model
from training.train import simple_train
from evaluation.validation import simple_validation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 4
num_epochs = 5
learning_rate = 0.001
SAVE_MODEL_PATH = 'models/resnet50cifar.pt'
LOAD_PRETRAINED = False

train_loader, test_loader = load_CIFAR10(train_subset_size=10000)
model_instance = ResNet50()

model = load_model(model_instance,
                   SAVE_MODEL_PATH) if LOAD_PRETRAINED else model_instance.to(device)

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
