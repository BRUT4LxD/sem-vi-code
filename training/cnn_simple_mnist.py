import torch
import torch.nn as nn
from architectures.sample_conv import ConvNetMNIST
from utils.dataset_loader import load_MNIST
from utils.io import load_model
from utils.train import simple_train
from utils.validation import simple_validation
from utils.visualization import simple_visualize
from constants.model_classes import mnist_classes

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

batch_size = 4
num_epochs = 5
learning_rate = 0.001
MODEL_SAVE_PATH = './models/cnn-mnist.pt'
LOAD_PRETRAINED = False

train_loader, test_loader = load_MNIST()

model_instance = ConvNetMNIST()

model = load_model(model_instance,
                   MODEL_SAVE_PATH) if LOAD_PRETRAINED else model_instance.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

simple_train(model=model,
             loss_fn=criterion,
             optimizer=optimizer,
             train_loader=train_loader,
             num_epochs=num_epochs,
             device=device,
             SAVE_MODEL_PATH=MODEL_SAVE_PATH)

simple_validation(model, test_loader, batch_size, mnist_classes, device=device)
