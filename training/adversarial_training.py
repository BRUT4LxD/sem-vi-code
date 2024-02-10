from typing import List
import torch
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.optim import Adam

from attacks.simple_attacks import SimpleAttacks
from config.imagenette_classes import ImageNetteClasses
from data_eng.attacked_dataset_generator import AttackedDatasetGenerator, AttackedDatasetGeneratorResult
from data_eng.dataset_loader import load_empty_dataloader, load_imagenette
from data_eng.io import save_model
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from domain.attack_distance_score import AttackDistanceScore

from evaluation.validation import Validation
import copy

class AdversarialTraining:
    def __init__(self, model: Module, learning_rate: float, attack_names: List['str'], model_name: str = None, device: str = 'gpu', key: str = None):
        self.model = model
        self.attack_names = attack_names
        self.learning_rate = learning_rate
        self.model_name = model_name if model_name is not None else model.__class__.__name__
        self.device = device
        self.key = key if key is not None else datetime.now().strftime("%Y-%m-%d_%H-%M")

    def _log_attack_distance_score(self, writer: SummaryWriter, attack_distance_score: AttackDistanceScore, iteration: int, prefix: str = 'train'):
        writer.add_scalar(f"Distance/{prefix}/L1", attack_distance_score.l1, iteration)
        writer.add_scalar(f"Distance/{prefix}/L2", attack_distance_score.l2, iteration)
        writer.add_scalar(f"Distance/{prefix}/Linf", attack_distance_score.linf, iteration)
        writer.add_scalar(f"Distance/{prefix}/power", attack_distance_score.power, iteration)

    def _generate_attacked_dataset(self, images_per_attack: int, attack_ratio: float, batch_size: int = 1, test_size: float = 0.5):
        self.model.eval()
        train_generator_result = AttackedDatasetGenerator.get_attacked_imagenette_dataset(
              model=self.model,
              model_name=self.model_name,
              batch_size=batch_size,
              shuffle=True,
              use_test_set=False,
              attack_names=self.attack_names,
              num_of_images_per_attack=images_per_attack,
              attack_ratio=attack_ratio)

        test_generator_result: AttackedDatasetGeneratorResult = AttackedDatasetGenerator.get_attacked_imagenette_dataset(
            model=self.model,
            model_name=self.model_name,
            batch_size=batch_size,
            shuffle=True,
            use_test_set=True,
            attack_names=self.attack_names,
            num_of_images_per_attack=int(images_per_attack * test_size),
            attack_ratio=attack_ratio)

        return train_generator_result, test_generator_result
    
    def _detach_dataloader(self, dataloader: DataLoader):
        for images, labels in dataloader:
            images = images.detach().to('cpu')
            labels = labels.detach().to('cpu')

    def train(self,
        epochs_per_iter: int,
        writer: SummaryWriter,
        iterations: int = 1,
        save_model_path: str = None,
        images_per_attack: int = 20,
        attack_ratio=0.2,
        test_size=0.2,
        batch_size: int = 1,
        stop_loss: float = 0.00001):

        criterion = CrossEntropyLoss()
        imagenette_to_imagenet_index_map = ImageNetteClasses.get_imagenette_to_imagenet_map_by_index()
        _, test_loader = load_imagenette(batch_size=128, shuffle=True)

        self.model.to(self.device)
        current_iteration = 0
        epoch = 0
        while current_iteration < iterations:
            try:
                train_generator_result, test_generator_result = self._generate_attacked_dataset(images_per_attack, attack_ratio, batch_size, test_size)
            except RuntimeError as e:
                print(f'[Adversarial Training] [Dataset Generation] Error: {e}')
                print(f'lowering images_per_attack by 10% to {images_per_attack - int(images_per_attack/10)} and try again')
                images_per_attack = images_per_attack - int(images_per_attack/10)
                if images_per_attack == 1:
                    break
                continue
            except Exception as e:
                print(f'[Adversarial Training] [Dataset Generation] Error: {e}')
                continue
            self.model.train()
            optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
            scheduler = StepLR(optimizer, step_size=int(epochs_per_iter/6) + 1, gamma=0.5)
            current_epoch = 0
            while current_epoch < epochs_per_iter:
                if current_epoch > 1:
                    current_loss = loss.item()
                    if current_loss < stop_loss:
                        print(f'Loss is less then {stop_loss}, stop training')
                        break
                try:
                    for images, labels in train_generator_result.data_loader:
                        for original, mapped in imagenette_to_imagenet_index_map.items():
                            mask = labels == original
                            labels[mask] = mapped

                        images = images.to(self.device)
                        labels = labels.to(self.device)

                        # Forward pass
                        optimizer.zero_grad()
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)

                        # Backward and optimize
                        loss.backward(retain_graph=True)
                        optimizer.step()

                        del images, labels
                        torch.cuda.empty_cache()
                except RuntimeError as e:
                    print(f'[Adversarial Training] Error during training: {e}')
                    print(f'lowering images_per_attack by 10% to {images_per_attack - int(images_per_attack/10)} and try again')
                    images_per_attack = images_per_attack - int(images_per_attack/10)
                    if images_per_attack == 1:
                        break
                    continue
                except Exception as e:
                    print(f'[Adversarial Training] Error during training: {e}')
                    continue

                print(f'iteration {current_iteration} epoch {current_epoch + 1}/{epochs_per_iter}, loss = {loss.item():.12f} lr={optimizer.param_groups[0]["lr"]}')
                writer.add_scalar("Loss/train", loss.item(), epoch)
                writer.add_scalar("Learning rate", optimizer.param_groups[0]['lr'], epoch)

                # validation
                val_result = Validation.validate_imagenet_with_imagenette_classes(model=self.model, model_name=self.model_name, test_loader=test_loader, device=self.device, print_results=False)
                adv_val_result = Validation.validate_imagenet_with_imagenette_classes(model=self.model, model_name=self.model_name, test_loader=test_generator_result.data_loader, device=self.device, print_results=False)
                writer.add_scalar("Accuracy/train", val_result.accuracy, epoch)
                writer.add_scalar("Accuracy/adv_train", adv_val_result.accuracy, epoch)

                print(f'validation accuracy: {val_result.accuracy:.4f}, adv validation accuracy: {adv_val_result.accuracy:.4f}')
                scheduler.step()
                current_epoch += 1
                epoch += 1

            current_iteration += 1
            self._log_attack_distance_score(writer, train_generator_result.attack_distance_score, current_iteration)
            self._log_attack_distance_score(writer, test_generator_result.attack_distance_score, current_iteration, prefix='test')

        print('Finished training')

        if save_model_path is not None:
            save_path_dir = os.path.dirname(save_model_path)
            if not os.path.exists(save_path_dir):
                os.makedirs(save_path_dir)
            save_model(self.model, save_model_path)

        writer.close()
        return val_result, adv_val_result

    def train_progressive(self,
        epochs_per_iter: int,
        writer: SummaryWriter,
        iterations: int = 1,
        save_model_path: str = None,
        images_per_attack: int = 10,
        attack_ratio=0.2,
        test_size=0.2,
        batch_size: int = 1,
        stop_loss: float = 0.00001):

        criterion = CrossEntropyLoss()
        imagenette_to_imagenet_index_map = ImageNetteClasses.get_imagenette_to_imagenet_map_by_index()
        _, test_loader = load_imagenette(batch_size=8, shuffle=True)

        self.model.to(self.device)
        current_iteration = 0
        epoch = 0
        train_attacked_loader, test_attacked_loader = load_empty_dataloader(), load_empty_dataloader()

        while current_iteration < iterations:
            try:
                train_generator_result, test_generator_result = self._generate_attacked_dataset(images_per_attack, attack_ratio, batch_size, test_size)
                train_attacked_loader = DataLoader(ConcatDataset([train_attacked_loader.dataset, train_generator_result.data_loader.dataset]), batch_size=batch_size, shuffle=True)
                test_attacked_loader = DataLoader(ConcatDataset([test_attacked_loader.dataset, test_generator_result.data_loader.dataset]), batch_size=batch_size, shuffle=True)
            except RuntimeError as e:
                print(f'[Adversarial Training] [Dataset Generation] Error: {e}')
                temp = images_per_attack - int(images_per_attack/10)
                images_per_attack = temp if temp < images_per_attack else images_per_attack - 1
                print(f'lowering images_per_attack by 10% to {images_per_attack - int(images_per_attack/10)} and try again')
                current_iteration += 1
                if images_per_attack == 1:
                    break
                continue
            except Exception as e:
                print(f'[Adversarial Training] [Dataset Generation] Error: {e}')
                current_iteration += 1
                continue

            self.model.train()
            optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
            scheduler = StepLR(optimizer, step_size=int(epochs_per_iter/6) + 1, gamma=0.5)
            current_epoch = 0

            print(f'train dataset size: {len(train_attacked_loader.dataset)}, test dataset size: {len(test_attacked_loader.dataset)}')
            while current_epoch < epochs_per_iter:
                if current_epoch > 1:
                    current_loss = loss.item()
                    if current_loss < stop_loss:
                        print(f'Loss is less then {stop_loss}, stop training')
                        break
                try:
                    for images, labels in train_attacked_loader:
                        for original, mapped in imagenette_to_imagenet_index_map.items():
                            mask = labels == original
                            labels[mask] = mapped

                        images = images.to(self.device)
                        labels = labels.to(self.device)

                        # Forward pass
                        optimizer.zero_grad()
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)

                        # Backward and optimize
                        loss.backward()
                        optimizer.step()

                except RuntimeError as e:
                    print(f'[Adversarial Training] Error during training: {e}')
                    print(f'lowering images_per_attack by 10% to {images_per_attack - int(images_per_attack/10)} and try again')
                    images_per_attack = images_per_attack - int(images_per_attack/10)
                    current_iteration += 1
                    if images_per_attack == 1:
                        break
                    continue
                except Exception as e:
                    print(f'[Adversarial Training] Error during training: {e}')
                    current_iteration += 1
                    if images_per_attack == 1:
                        break
                    continue

                print(f'iteration {current_iteration} epoch {current_epoch + 1}/{epochs_per_iter}, loss = {loss.item():.12f} lr={optimizer.param_groups[0]["lr"]}')
                writer.add_scalar("Loss/train", loss.item(), epoch)
                writer.add_scalar("Learning rate", optimizer.param_groups[0]['lr'], epoch)

                # validation
                val_result = Validation.validate_imagenet_with_imagenette_classes(model=self.model, model_name=self.model_name, test_loader=test_loader, device=self.device, print_results=False)
                adv_val_result = Validation.validate_imagenet_with_imagenette_classes(model=self.model, model_name=self.model_name, test_loader=test_attacked_loader, device=self.device, print_results=False)
                writer.add_scalar("Accuracy/train", val_result.accuracy, epoch)
                writer.add_scalar("Accuracy/adv_train", adv_val_result.accuracy, epoch)

                print(f'validation accuracy: {val_result.accuracy:.4f}, adv validation accuracy: {adv_val_result.accuracy:.4f}')
                scheduler.step()
                current_epoch += 1
                epoch += 1

            # self._detach_dataloader(train_attacked_loader)
            # self._detach_dataloader(test_attacked_loader)
            # self._detach_dataloader(test_loader)

            current_iteration += 1
            self._log_attack_distance_score(writer, train_generator_result.attack_distance_score, current_iteration)
            self._log_attack_distance_score(writer, test_generator_result.attack_distance_score, current_iteration, prefix='test')

        print('Finished training')

        if save_model_path is not None:
            save_path_dir = os.path.dirname(save_model_path)
            if not os.path.exists(save_path_dir):
                os.makedirs(save_path_dir)
            save_model(self.model, save_model_path)

        writer.close()
        return val_result, adv_val_result
