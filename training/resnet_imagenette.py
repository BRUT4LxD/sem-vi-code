import torch
import torch.nn as nn
from architectures.resnet import ResNet18
from data_eng.dataset_loader import load_imagenette
from constants.model_classes import imagenette_classes
from data_eng.io import load_model
from training.train import simple_train
from evaluation.validation import simple_validation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 64
num_epochs = 5
learning_rate = 0.001
SAVE_MODEL_PATH = 'models/resnet18imagenette.pt'
LOAD_PRETRAINED = False


model_instance = ResNet18()
model = load_model(model_instance,
                   SAVE_MODEL_PATH) if LOAD_PRETRAINED else model_instance.to(device)

train_loader, test_loader = load_imagenette()

model = ResNet18().to(device)

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
                  classes=imagenette_classes,
                  device=device)
