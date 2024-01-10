from typing import List
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from attacks.simple_attacks import SimpleAttacks
from config.imagenet_classes import ImageNetClasses
from config.imagenet_models import ImageNetModels
from config.imagenette_classes import ImageNetteClasses
from data_eng.attacked_dataset_generator import AttackedDatasetGenerator
from data_eng.dataset_loader import DatasetLoader, DatasetType
from domain.attack_eval_score import AttackEvaluationScore
from evaluation.validation import Validation
import os
import csv

class AdversarialValidationAccuracyResult:
    def __init__(self, model_name: str, accuracy: float, accuracies: List['float'], class_names: List['str'] , attack_ratio: float):
        self.model_name = model_name
        self.accuracy = round(accuracy,2)
        self.accuracies = [round(acc, 2) for acc in accuracies]
        self.class_names = class_names
        self.attack_ratio = attack_ratio

    def __str__(self):
        return f'{self.model_name}: {round(self.accuracy, 2)}%'

    def save_csv(self, it: str, save_path):
        headers = ['it'].extend(self.class_names)
        results = [it].extend(self.accuracies)

        with open(save_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerow(results)

    def append_to_csv_file(self, it: str, save_path):
        results = [it].extend(self.accuracies)

        with open(save_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(results)

class AdversarialValidation:

    @staticmethod
    def validate_with_imagenette(model: torch.nn.Module, attack_names: List['str'], model_name=None, device='gpu', save_path=None, print_results=True, test_subset_size: int = 1000, batch_size: int = 1) -> AdversarialValidationAccuracyResult:
        _, test_imagenette_loader = DatasetLoader.get_dataset_by_type(dataset_type=DatasetType.IMAGENETTE, batch_size=batch_size, test_subset_size=test_subset_size)
        images_per_attack = int(len(test_imagenette_loader.dataset) / len(attack_names))
        attacked_results: DataLoader = AttackedDatasetGenerator.get_attacked_imagenette_dataset(
            model=model,
            attack_names=attack_names,
            attack_ratio=1.0,
            num_of_images_per_attack=images_per_attack,
            batch_size=1,
            use_test_set=True,
            model_name=model_name
        )

        merged_dataset = ConcatDataset([test_imagenette_loader.dataset, attacked_results.dataset])
        test_loader = DataLoader(merged_dataset, batch_size=batch_size, shuffle=True)

        results =  Validation.validate_imagenet_with_imagenette_classes(
            model=model,
            test_loader=test_loader,
            device=device,
            model_name=model_name,
            print_results=print_results,
            save_path=save_path
        )

        return AdversarialValidationAccuracyResult(results.model_name, results.accuracy, results.accuracies, ImageNetteClasses.get_classes(), 0.5)

    @staticmethod
    def validate_with_imagenette_multimodel(
        model: torch.nn.Module,
        attack_names: List['str'],
        attack_model_names: List['str'],
        batch_size: int = 1,
        model_name=None,
        device='gpu',
        save_path=None,
        print_results=True,
        test_subset_size: int = 1000):

        _, test_imagenette_loader = DatasetLoader.get_dataset_by_type(dataset_type=DatasetType.IMAGENETTE, batch_size=batch_size, test_subset_size=test_subset_size)
        images_per_attack = int(len(test_imagenette_loader.dataset) / (len(attack_names) * len(attack_model_names)))
        print(f'Images per attack: {images_per_attack}')
        if images_per_attack == 0:
            raise Exception('Not enough images per attack')

        attack_results = []

        for attack_model_name in attack_model_names:
            attack_model = ImageNetModels.get_model(attack_model_name)
            attacked_result: DataLoader = AttackedDatasetGenerator.get_attacked_imagenette_dataset(
                model=attack_model,
                attack_names=attack_names,
                attack_ratio=1.0,
                num_of_images_per_attack=images_per_attack,
                batch_size=1,
                use_test_set=True,
                model_name=model_name
            )

            attack_results.append(attacked_result.dataset)

        merged_dataset = ConcatDataset([test_imagenette_loader.dataset, *attack_results])
        test_loader = DataLoader(merged_dataset, batch_size=batch_size, shuffle=True)

        results =  Validation.validate_imagenet_with_imagenette_classes(
            model=model,
            test_loader=test_loader,
            device=device,
            model_name=model_name,
            print_results=print_results,
            save_path=save_path
        )

        return AdversarialValidationAccuracyResult(results.model_name, results.accuracy, results.accuracies, ImageNetteClasses.get_classes(), 0.5)


