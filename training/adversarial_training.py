from typing import List
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from attacks.simple_attacks import SimpleAttacks
from config.imagenette_classes import ImageNetteClasses
from data_eng.attacked_dataset_generator import AttackedDatasetGenerator
from data_eng.io import save_model
import os
from torch.utils.tensorboard import SummaryWriter

class AdversarialTraining:
    def __init__(self, model: Module, optimizer: Optimizer, attack_names: List['str'], model_name: str = None, device: str = 'gpu'):
        self.model = model
        self.optimizer = optimizer
        self.attack_names = attack_names
        self.model_name = model_name if model_name is not None else model.__class__.__name__
        self.device = device

    def train(self, num_epochs=10, save_model_path: str = None, images_per_attack: int = 20, attack_ratio=0.2, batch_size: int = 1):
        criterion = CrossEntropyLoss()
        n_total_steps = len(self.attack_names) * images_per_attack
        imagenette_to_imagenet_index_map = ImageNetteClasses.get_imagenette_to_imagenet_map_by_index()
        train_loader = AttackedDatasetGenerator.get_attacked_imagenette_dataset(
              model=self.model,
              model_name=self.model_name,
              batch_size=batch_size,
              shuffle=True,
              use_test_set=False,
              attack_names=self.attack_names,
              num_of_images_per_attack=images_per_attack,
              attack_ratio=attack_ratio)
        self.model.train()
        self.model.to(self.device)
        writer = SummaryWriter(filename_suffix=f'_{self.model_name}_adversarial_training')
        for epoch in range(num_epochs):
            i = 0
            for images, labels in train_loader:
                for original, mapped in imagenette_to_imagenet_index_map.items():
                    mask = labels == original
                    labels[mask] = mapped

                i += labels.shape[0]
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                writer.add_scalar("Loss/train", loss.item(), epoch)

                # Backward and optimize
                loss.backward(retain_graph=True)
                self.optimizer.step()

                if (i + 1) % 2000 == 0:
                    print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.4f}')

            writer.add_scalar("Loss/train", loss, epoch)
            print(f'epoch {epoch + 1}/{num_epochs}, loss = {loss.item():.4f} lr = ')

        print('Finished training')

        if save_model_path is not None:
            save_path_dir = os.path.dirname(save_model_path)
            if not os.path.exists(save_path_dir):
                os.makedirs(save_path_dir)
            save_model(self.model, save_model_path)