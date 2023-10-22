import torch
import torch.nn as nn
from architectures.resnet import ResNet101, ResNet152, ResNet18, ResNet50
from data_eng.dataset_loader import load_imagenette
from config.model_classes import imagenette_classes
from data_eng.io import load_model
from evaluation.validation import Validation
from training.train import simple_train


def train_all_resnet(num_epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = [ResNet18(), ResNet50(), ResNet101(), ResNet152()]
    paths = ['models/resnet18imagenette.pt', 'models/resnet50imagenette.pt',
             'models/resnet101imagenette.pt', 'models/resnet152imagenette.pt']
    batch_sizes = [64, 64, 64, 32]

    learning_rate = 0.001
    for model, path, batch_size in zip(models, paths, batch_sizes):
        torch.cuda.empty_cache()
        model.to(device)
        train_loader, test_loader = load_imagenette(batch_size=batch_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        simple_train(model=model,
                     loss_fn=criterion,
                     optimizer=optimizer,
                     train_loader=train_loader,
                     num_epochs=num_epochs,
                     device=device,
                     SAVE_MODEL_PATH=path)

        Validation.simple_validation(model=model,
                          test_loader=test_loader,
                          classes=imagenette_classes,
                          device=device)


def validate_all_resnet():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = [ResNet18(), ResNet50(), ResNet101(), ResNet152()]
    paths = ['models/resnet18imagenette.pt', 'models/resnet50imagenette.pt',
             'models/resnet101imagenette.pt', 'models/resnet152imagenette.pt']
    _, test_loader = load_imagenette(batch_size=5)
    for model, path in zip(models, paths):
        torch.cuda.empty_cache()
        print(path)
        try:
            model = load_model(model, path).to(device)
            Validation.simple_validation(model=model,
                              test_loader=test_loader,
                              classes=imagenette_classes,
                              device=device)
        except Exception as e:
            print("Error", e)
