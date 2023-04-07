import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from architectures.dumb import ConvNetCIFAR
from utils.dataset_loader import load_CIFAR10
from utils.io import load_model
from utils.train import simple_train
from utils.validation import simple_validation
from utils.visualization import simple_visualize
from constants.model_classes import cifar_classes

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 4
num_epochs = 5
learning_rate = 0.001
MODEL_SAVE_PATH = './models/cnn-cifar.pth'
LOAD_PRETRAINED = False

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

train_dataset, test_dataset = load_CIFAR10(transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)

model = load_model(
    MODEL_SAVE_PATH) if LOAD_PRETRAINED else ConvNetCIFAR().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

simple_train(model=model,
             loss_fn=criterion,
             optimizer=optimizer,
             train_loader=train_loader,
             num_epochs=num_epochs,
             device=device,
             SAVE_MODEL_PATH=MODEL_SAVE_PATH)

simple_validation(model, test_loader, batch_size, cifar_classes, device=device)
simple_visualize(model, test_loader, batch_size, cifar_classes, device=device)
