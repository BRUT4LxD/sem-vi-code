from typing import List
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from attacks.simple_attacks import SimpleAttacks
from config.imagenette_classes import ImageNetteClasses
from data_eng.attacked_dataset_generator import AttackedDatasetGenerator, AttackedDatasetGeneratorResult
from data_eng.dataset_loader import load_imagenette
from data_eng.io import save_model
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from domain.attack_distance_score import AttackDistanceScore

from evaluation.validation import Validation

class AdversarialTraining:
    def __init__(self, model: Module, optimizer: Optimizer, attack_names: List['str'], model_name: str = None, device: str = 'gpu', key: str = None):
        self.model = model
        self.optimizer = optimizer
        self.attack_names = attack_names
        self.model_name = model_name if model_name is not None else model.__class__.__name__
        self.device = device
        self.key = key if key is not None else datetime.now().strftime("%Y-%m-%d_%H-%M")

    def train(self, num_epochs=10, save_model_path: str = None, images_per_attack: int = 20, attack_ratio=0.2, batch_size: int = 1, it_num: int = 0, include_validation: bool = True):
        criterion = CrossEntropyLoss()
        n_total_steps = len(self.attack_names) * images_per_attack
        imagenette_to_imagenet_index_map = ImageNetteClasses.get_imagenette_to_imagenet_map_by_index()
        train_generator_result = AttackedDatasetGenerator.get_attacked_imagenette_dataset(
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

        lr = self.optimizer.param_groups[0]['lr']
        writer = SummaryWriter(log_dir=f'runs/adversarial_training/{self.model_name}/{self.key}it={it_num}_lr={lr}_ar={attack_ratio}_imgs={images_per_attack*len(self.attack_names)}', )

        scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=5, verbose=True, eps=1e-4, factor=0.5, min_lr=1e-7)
        if include_validation:
            _, test_loader = load_imagenette(batch_size=128, shuffle=True)
            test_generator_result: AttackedDatasetGeneratorResult = AttackedDatasetGenerator.get_attacked_imagenette_dataset(
              model=self.model,
              model_name=self.model_name,
              batch_size=batch_size,
              shuffle=True,
              use_test_set=True,
              attack_names=self.attack_names,
              num_of_images_per_attack=images_per_attack,
              attack_ratio=attack_ratio)

            writer.add_hparams(hparam_dict={
                "lr": lr,
                "attack_ratio": attack_ratio,
                "images_per_attack": images_per_attack,
                "batch_size": batch_size,
                "it_num": it_num,
                "attack_names": ','.join(self.attack_names),
            }, metric_dict={
                'Test avg L1 ': test_generator_result.attack_distance_score.l1,
                'Test avg L2 ': test_generator_result.attack_distance_score.l2,
                'Test avg Linf ': test_generator_result.attack_distance_score.linf,
                'Test avg power ': test_generator_result.attack_distance_score.power,
            })

        writer.add_hparams(hparam_dict={
                "lr": lr,
                "attack_ratio": attack_ratio,
                "images_per_attack": images_per_attack,
                "batch_size": batch_size,
                "it_num": it_num,
                "attack_names": ','.join(self.attack_names),
        }, metric_dict= {
            'Train avg L1 ': train_generator_result.attack_distance_score.l1,
            'Train avg L2 ': train_generator_result.attack_distance_score.l2,
            'Train avg Linf ': train_generator_result.attack_distance_score.linf,
            'Train avg power ': train_generator_result.attack_distance_score.power,
        })

        for epoch in range(num_epochs):
            if epoch > 1:
                # get current loss and if its less then 0.0000001 then stop training
                current_loss = loss.item()
                if current_loss < 0.0000001:
                    print('Loss is less then 0.0000001, stop training')
                    break
            i = 0
            for images, labels in train_generator_result.data_loader:
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

                # Backward and optimize
                loss.backward(retain_graph=True)
                self.optimizer.step()

                if (i + 1) % 2000 == 0:
                    print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.12f}')

            print(f'epoch {epoch + 1}/{num_epochs}, loss = {loss.item():.12f} lr={self.optimizer.param_groups[0]["lr"]}')
            writer.add_scalar("Loss/train", loss.item(), epoch)
            writer.add_scalar("Learning rate", self.optimizer.param_groups[0]['lr'], epoch)
            if include_validation:
                val_result = Validation.validate_imagenet_with_imagenette_classes(model=self.model, model_name=self.model_name, test_loader=test_loader, device=self.device, print_results=False)
                adv_val_result = Validation.validate_imagenet_with_imagenette_classes(model=self.model, model_name=self.model_name, test_loader=test_generator_result.data_loader, device=self.device, print_results=False)
                writer.add_scalar("Accuracy/train", val_result.accuracy, epoch)
                writer.add_scalar("Accuracy/adv_train", adv_val_result.accuracy, epoch)

            scheduler.step(loss)
        print('Finished training')

        if save_model_path is not None:
            save_path_dir = os.path.dirname(save_model_path)
            if not os.path.exists(save_path_dir):
                os.makedirs(save_path_dir)
            save_model(self.model, save_model_path)

        writer.close()
        return val_result, adv_val_result