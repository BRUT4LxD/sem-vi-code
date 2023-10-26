import torch
import torch.nn as nn
from architectures.vgg import VGG, VGG11, VGG13, VGG16, VGG19
from data_eng.dataset_loader import load_imagenette
from config.imagenette_classes import ImageNetteClasses
from evaluation.validation import Validation
from training.train import Training


def train_all_vgg(num_epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rate = 0.001
    models = [VGG11(), VGG13(), VGG16(), VGG19()]
    paths = ['models/vgg11imagenette.pt', 'models/vgg13imagenette.pt',
             'models/vgg16imagenette.pt', 'models/vgg19imagenette.pt']
    batch_sizes = [64, 64, 32, 32]
    for model, path, batch_size in zip(models, paths, batch_sizes):
        train_loader, test_loader = load_imagenette(batch_size=batch_size)
        torch.cuda.empty_cache()
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        Training.simple_train(model=model,
                     loss_fn=criterion,
                     optimizer=optimizer,
                     train_loader=train_loader,
                     num_epochs=num_epochs,
                     device=device,
                     SAVE_MODEL_PATH=path)

        Validation.simple_validation(model=model,
                          test_loader=test_loader,
                          classes=ImageNetteClasses.get_classes(),
                          device=device)

        del train_loader, test_loader, model, criterion, optimizer
