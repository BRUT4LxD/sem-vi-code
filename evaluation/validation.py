from typing import List
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from config.imagenet_models import ImageNetModels
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
    def validate_model_by_name(model_name: str, test_loader: DataLoader, classes, device: str):
        model = ImageNetModels.get_model(model_name)
        model.to(device)
        Validation.simple_validation(model, test_loader, classes, device)