from datetime import datetime
from data_eng.io import save_model
import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from evaluation.validation import Validation
class Training:

    @staticmethod
    def simple_train(
        model: torch.nn.Module,
        loss_fn: torch.nn.CrossEntropyLoss,
        optimizer: torch.optim.Adam,
        train_loader: DataLoader,
        test_loader: DataLoader = None,
        num_epochs: int = 5,
        device: str='gpu',
        SAVE_MODEL_PATH: str = None,
        model_name: str = None,
        writer: SummaryWriter = None):

        n_total_steps = len(train_loader)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.8)
        model_name = model_name if model_name is not None else model.__class__.__name__
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss: torch.Tensor = loss_fn(outputs, labels)

                loss.backward()
                optimizer.step()

                if (i + 1) % 2000 == 0:
                    print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.8f}')

            scheduler.step()
            print(f'({model_name}) epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.8f} lr = {scheduler.get_last_lr()[0]}')
            writer.add_scalar("Loss/train", loss.item(), epoch)
            writer.add_scalar("Learning rate", optimizer.param_groups[0]['lr'], epoch)

            if test_loader is not None:
                res = Validation.validate_imagenet_with_imagenette_classes(
                    model=model,
                    test_loader=test_loader,
                    model_name=model_name,
                    device=device,
                    print_results=False
                )
                writer.add_scalar("Accuracy/train", res.accuracy, epoch)
                _ = model.train()
                
            if loss.item() < 0.00000001:
                break

        print('Finished training')

        if SAVE_MODEL_PATH is not None:
            save_model(model, SAVE_MODEL_PATH)

    @staticmethod
    def train_imagenette(
        model: torch.nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader = None,
        learning_rate=0.001,
        num_epochs=5,
        device='cuda',
        save_model_path=None,
        model_name=None,
        writer: SummaryWriter = None):

        if writer is None:
            date = datetime.now().strftime("%d-%m-%Y_%H-%M")
            writer = SummaryWriter(log_dir=f'runs/imagenette_training/{model_name}/{date}_lr={learning_rate}')

        _ = model.train()
        _ = model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        Training.simple_train(model, criterion, optimizer, train_loader, test_loader, num_epochs, device, save_model_path, model_name, writer)
