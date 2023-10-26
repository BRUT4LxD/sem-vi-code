from data_eng.io import save_model
import torch
from torch.optim.lr_scheduler import StepLR

class Training:

    @staticmethod
    def simple_train(model: torch.nn.Module, loss_fn, optimizer: torch.optim.Adam, train_loader, num_epochs=5, device='gpu', SAVE_MODEL_PATH=None, model_name=None):
        n_total_steps = len(train_loader)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.2)
        model_name = model_name if model_name is not None else model.__class__.__name__
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)

                loss.backward()
                optimizer.step()

                if (i + 1) % 2000 == 0:
                    print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.4f}')

            scheduler.step()
            print(f'({model_name}) epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.4f} lr = {scheduler.get_last_lr()[0]}')

        print('Finished training')

        if SAVE_MODEL_PATH is not None:
            save_model(model, SAVE_MODEL_PATH)

    @staticmethod
    def train_imagenette(model: torch.nn.Module, train_loader, learning_rate=0.001, num_epochs=5, device='gpu', save_model_path=None, model_name=None):

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        Training.simple_train(model, criterion, optimizer, train_loader, num_epochs, device, save_model_path, model_name)
