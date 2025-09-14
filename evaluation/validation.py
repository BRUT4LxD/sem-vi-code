from typing import List
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from config.imagenet_classes import ImageNetClasses
from config.imagenet_models import ImageNetModels
from config.imagenette_classes import ImageNetteClasses
from domain.attack.attack_eval_score import AttackEvaluationScore
import os
import csv

class ValidationAccuracyResult:
    def __init__(self, model_name: str, accuracy: float, accuracies: List['float'], class_names: List['str']):
        self.model_name = model_name
        self.accuracy = round(accuracy, 2)
        self.accuracies = [round(acc, 2) for acc in accuracies]
        self.class_names = class_names

    def __str__(self):
        return f'{self.model_name}: {round(self.accuracy, 2)}%'

    def save_csv(self, it: str, save_path: str):
        dirname = os.path.dirname(save_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        headers = ['it', 'acc'] +  self.class_names
        results = [it, self.accuracy] + self.accuracies

        with open(save_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerow(results)

    def append_to_csv_file(self, it: str, save_path: str):
        results = [it, self.accuracy] + self.accuracies

        dirname = os.path.dirname(save_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        if not os.path.exists(save_path):
            headers = ['it', 'acc'] + self.class_names
            with open(save_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
                file.close()

        with open(save_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(results)

class ValidationAccuracyBinaryResult:
    def __init__(self, model_name: str, accuracy: float):
        self.model_name = model_name
        self.accuracy = round(accuracy, 2)

    def __str__(self):
        return f'{self.model_name}: {round(self.accuracy, 2)}%'

    def save_csv(self, it: str, save_path: str):
        dirname = os.path.dirname(save_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        headers = ['it', 'acc']
        results = [it, self.accuracy]

        with open(save_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerow(results)

    def append_to_csv_file(self, it: str, save_path: str):
        results = [it, self.accuracy]

        dirname = os.path.dirname(save_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        if not os.path.exists(save_path):
            headers = ['it', 'acc']
            with open(save_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
                file.close()

        with open(save_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(results)

class Validation:

    @staticmethod
    @torch.no_grad()
    def simple_validation(model: torch.nn.Module, test_loader: DataLoader, classes: list, device='cuda'):
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(10)]
        n_class_samples = [0 for i in range(10)]

        model.eval()
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predictions == labels).sum().item()

            for i in range(len(labels)):
                label = labels[i]
                pred = predictions[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        print(f'accuracy = {acc}%')

        for i in range(10):
            if n_class_samples[i] == 0:
                print(f'accuracy of {classes[i]}: not enough samples')
                continue
            class_acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'accuracy of {classes[i]}: {class_acc}%')

    @staticmethod
    @torch.no_grad()
    def validate_imagenet_with_imagenette_classes(
        model: torch.nn.Module,
        test_loader: DataLoader,
        model_name=None,
        device='cuda',
        save_path=None,
        print_results=True
        ) -> ValidationAccuracyResult:
        imagenet_to_imagenette_class_map = ImageNetClasses.get_imagenet_to_imagenette_index_map()
        imagenette_classes = ImageNetteClasses.get_classes()
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(len(imagenet_to_imagenette_class_map))]
        n_class_samples = [0 for i in range(len(imagenet_to_imagenette_class_map))]

        model.eval()
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            # Apply the mapping to the predictions
            predictions = predictions.clone()
            for original, mapped in imagenet_to_imagenette_class_map.items():
                mask = predictions == original
                predictions[mask] = mapped

            n_samples += labels.size(0)
            n_correct += (predictions == labels).sum().item()

            for i in range(len(labels)):
                label = labels[i]
                pred = predictions[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        string_builder = ''
        string_builder += f'accuracy = {acc}%' + '\n'

        for i in range(10):
            if n_class_samples[i] == 0:
                string_builder += f'{imagenette_classes[i]}: not enough samples' + '\n'
                continue
            class_acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            string_builder += f'{imagenette_classes[i]}: {class_acc}%' + '\n'

        if print_results:
            print(string_builder)

        if save_path is not None:
            save_path_dir = os.path.dirname(save_path)
            if not os.path.exists(save_path_dir):
                os.makedirs(save_path_dir)
            with open(save_path, 'w') as f:
                f.write(string_builder)

        model_name = model_name if model_name is not None else model.__class__.__name__

        accs = []
        for i in range(len(imagenet_to_imagenette_class_map)):
            if n_class_samples[i] == 0:
                accs.append(-1)
                continue
            accs.append(100.0 * n_class_correct[i] / n_class_samples[i])
        return ValidationAccuracyResult(model_name=model_name, accuracy=acc, accuracies=accs, class_names=imagenette_classes)


    @staticmethod
    @torch.no_grad()
    def validate_binary_classification(
        model: torch.nn.Module,
        test_loader: DataLoader,
        device='cuda',
        print_results=True,
        model_name=None
    ) -> ValidationAccuracyResult:
        n_correct = 0
        n_samples = 0

        model.eval()
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = torch.sigmoid(outputs) > 0.5 

            n_samples += labels.size(0)
            n_correct += (predictions.squeeze().int() == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        string_builder = f'Accuracy = {acc}%\n'

        if print_results:
            print(string_builder)

        model_name = model_name if model_name is not None else model.__class__.__name__
        return ValidationAccuracyBinaryResult(model_name=model_name, accuracy=acc)

    @staticmethod
    @torch.no_grad()
    def validate_imagenette_epoch(
        model: torch.nn.Module,
        test_loader: DataLoader,
        criterion,
        device='cuda',
        verbose=True,
        epoch=None,
        num_epochs=None
    ) -> dict:
        """
        Validate model for one epoch during ImageNette training.
        
        Args:
            model: PyTorch model to validate
            test_loader: Validation data loader
            criterion: Loss function
            device: Device to run validation on
            verbose: Whether to show progress bar
            epoch: Current epoch number (for progress display)
            num_epochs: Total number of epochs (for progress display)
            
        Returns:
            dict: Dictionary containing validation metrics (loss, accuracy)
        """
        model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", 
                              disable=not verbose)
            
            for data, target in progress_bar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct_predictions += pred.eq(target).sum().item()
                total_samples += target.size(0)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * correct_predictions / total_samples:.2f}%'
                })
        
        avg_loss = total_loss / len(test_loader)
        accuracy = 100. * correct_predictions / total_samples
        
        return {'loss': avg_loss, 'accuracy': accuracy}