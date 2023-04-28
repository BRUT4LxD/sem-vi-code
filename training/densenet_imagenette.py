import torch
import torch.nn as nn
from architectures.densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201
from data_eng.dataset_loader import load_imagenette
from config.model_classes import imagenette_classes
from training.train import simple_train
from evaluation.validation import simple_validation


def train_all_densenet(num_epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rate = 0.001
    models = [DenseNet121(), DenseNet161(), DenseNet169(), DenseNet201()]
    paths = ['models/densenet121imagenette.pt', 'models/densenet161imagenette.pt',
             'models/densenet169imagenette.pt', 'models/densenet201imagenette.pt']
    batch_sizes = [128, 64, 64, 32]
    i = -1
    for model, path, batch_size in zip(models, paths, batch_sizes):
        i += 1
        if i == 0:
            continue
        train_loader, test_loader = load_imagenette(batch_size=batch_size)
        torch.cuda.empty_cache()
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        simple_train(model=model,
                     loss_fn=criterion,
                     optimizer=optimizer,
                     train_loader=train_loader,
                     num_epochs=num_epochs,
                     device=device,
                     SAVE_MODEL_PATH=path)

        simple_validation(model=model,
                          test_loader=test_loader,
                          classes=imagenette_classes,
                          device=device)

        del train_loader, test_loader, model, criterion, optimizer
