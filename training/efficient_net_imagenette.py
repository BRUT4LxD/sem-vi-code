import torch
import torch.nn as nn
from architectures.efficientnet import EfficientNetB0
from data_eng.dataset_loader import load_imagenette
from config.model_classes import imagenette_classes
from training.train import simple_train
from evaluation.validation import simple_validation


def train_all_efficient_net(num_epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rate = 0.001
    models = [EfficientNetB0()]
    paths = ['models/efficientnetb0imagenette.pt']
    batch_sizes = [8]
    for model, path, batch_size in zip(models, paths, batch_sizes):
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

