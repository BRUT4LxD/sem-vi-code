from typing import List
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from config.imagenet_classes import ImageNetClasses
from config.imagenet_models import ImageNetModels
from config.imagenette_classes import ImageNetteClasses
from domain.attack_eval_score import AttackEvaluationScore

class Validation:

    @staticmethod
    @torch.no_grad()
    def simple_validation(model: torch.nn.Module, test_loader: DataLoader, classes: list, device='gpu'):
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
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'accuracy of {classes[i]}: {acc}%')

    @staticmethod
    @torch.no_grad()
    def validate_imagenet_with_imagenette_classes(model: torch.nn.Module, test_loader: DataLoader, device='gpu'):
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
        print(f'accuracy = {acc}%')

        for i in range(10):
            if n_class_samples[i] == 0:
                print(f'accuracy of {imagenette_classes[i]}: not enough samples')
                continue
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'accuracy of {imagenette_classes[i]}: {acc}%')
