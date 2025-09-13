from datetime import datetime
from data_eng.io import save_model
import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from evaluation.validation import Validation, ValidationAccuracyResult
from training.transfer.setup_pretraining import SetupPretraining

from torch.nn import Module, CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from torch.optim import Adam
from tqdm import tqdm

class Training:

    @staticmethod
    def simple_train(
        model: Module,
        loss_fn,
        optimizer: Adam,
        train_loader: DataLoader,
        test_loader: DataLoader = None,
        num_epochs: int = 5,
        device: str= 'cuda',
        save_model_path: str = None,
        model_name: str = None,
        writer: SummaryWriter = None,
        is_binary_classification: bool = False,
        stop_at_loss: float = None):

        n_total_steps = len(train_loader)
        stop_at_loss = stop_at_loss if stop_at_loss is not None else 0.00000001
        scheduler = StepLR(optimizer, step_size=5, gamma=0.8)
        model_name = model_name if model_name is not None else model.__class__.__name__
        for epoch in range(num_epochs):
            i = 0
            for images, labels in tqdm(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                if is_binary_classification:
                    outputs = outputs.squeeze(1)
                    labels = labels.float()

                loss: torch.Tensor = loss_fn(outputs, labels)

                loss.backward()
                optimizer.step()

                if (i + 1) % 2000 == 0:
                    print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.8f}')

                i = i + 1

            scheduler.step()
            print(f'({model_name}) epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.8f} lr = {scheduler.get_last_lr()[0]}')
            writer.add_scalar("Loss/train", loss.item(), epoch)
            writer.add_scalar("Learning rate", optimizer.param_groups[0]['lr'], epoch)

            if test_loader is not None:
                res: ValidationAccuracyResult

                if is_binary_classification:
                    res = Validation.validate_binary_classification(model, test_loader, device, False, model_name)
                else:
                    res = Validation.validate_imagenet_with_imagenette_classes(
                    model=model,
                    test_loader=test_loader,
                    model_name=model_name,
                    device=device,
                    print_results=False
                )
                writer.add_scalar("Accuracy/train", res.accuracy, epoch)
                _ = model.train()

            if loss.item() < stop_at_loss:
                break

        print('Finished training')

        if save_model_path is not None:
            save_model(model, save_model_path)

    @staticmethod
    def train_imagenette(
        model: Module,
        train_loader: DataLoader,
        test_loader: DataLoader = None,
        learning_rate=0.001,
        num_epochs=5,
        device='cuda',
        save_model_path=None,
        model_name=None,
        writer: SummaryWriter = None,
        setup_model=True):

        if writer is None:
            date = datetime.now().strftime("%d-%m-%Y_%H-%M")
            writer = SummaryWriter(log_dir=f'runs/imagenette_training/{model_name}/{date}_lr={learning_rate}')

        # Setup model for ImageNette if requested
        if setup_model:
            print(f"⚙️ Setting up {model_name} for ImageNette training...")
            model = SetupPretraining.setup_imagenette(model)
            print(f"✅ Model setup complete - ready for ImageNette (10 classes)")

        _ = model.train()
        _ = model.to(device)
        criterion = CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=learning_rate)

        Training.simple_train(model, criterion, optimizer, train_loader, test_loader, num_epochs, device, save_model_path, model_name, writer)

    @staticmethod
    def train_binary(
        model: Module,
        train_loader: DataLoader,
        test_loader: DataLoader = None,
        learning_rate=0.00001,
        num_epochs=5,
        device='cuda',
        save_model_path=None,
        model_name=None,
        writer: SummaryWriter = None):

        if writer is None:
            date = datetime.now().strftime("%d-%m-%Y_%H-%M")
            writer = SummaryWriter(log_dir=f'runs/binary_training/{model_name}/{date}_lr={learning_rate}')

        if save_model_path is None:
            save_model_path = f'./models/binary/{model_name}.pt'

        _ = model.train()
        _ = model.to(device)
        criterion = BCEWithLogitsLoss()
        optimizer = Adam(model.parameters(), lr=learning_rate)

        Training.simple_train(
            model=model,
            loss_fn=criterion,
            optimizer = optimizer,
            train_loader=train_loader,test_loader=test_loader,
            num_epochs=num_epochs,
            device=device,
            save_model_path=save_model_path,
            model_name=model_name,
            writer=writer,
            is_binary_classification=True,
            stop_at_loss=0.0000000001)
