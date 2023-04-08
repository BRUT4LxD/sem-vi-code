import torch
import torch.nn as nn
import torchvision.transforms as transforms
from architectures.sample_conv import ConvNetImageNet
from utils.dataset_loader import load_imagenette
from utils.io import load_model
from utils.train import simple_train
from utils.validation import simple_validation
from utils.visualization import simple_visualize
from constants.model_classes import imagenette_classes

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

batch_size = 4
num_epochs = 10
learning_rate = 0.001
MODEL_SAVE_PATH = './models/cnn-imagenette.pth'
LOAD_PRETRAINED = False

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset, test_dataset = load_imagenette(transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)

model = load_model(
    MODEL_SAVE_PATH) if LOAD_PRETRAINED else ConvNetImageNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

simple_train(model=model,
             loss_fn=criterion,
             optimizer=optimizer,
             train_loader=train_loader,
             num_epochs=num_epochs,
             device=device,
             SAVE_MODEL_PATH=MODEL_SAVE_PATH)

simple_validation(model, test_loader, batch_size,
                  imagenette_classes, device=device)

simple_visualize(model, test_loader, batch_size,
                 imagenette_classes, device=device)
