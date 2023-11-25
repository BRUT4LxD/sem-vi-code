import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import List
import os
from typing import List
from attacks.attack import Attack
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from attacks.attack_names import AttackNames
from config.imagenet_classes import ImageNetClasses
from config.imagenet_models import ImageNetModels
from config.imagenette_classes import ImageNetteClasses
from data_eng.dataset_loader import DatasetLoader
from datetime import datetime

from shared.model_utils import ModelUtils

class Transferability():

    @staticmethod
    def transferability_attack(
            attacked_model: torch.nn.Module,
            trans_models: List['torch.nn.Module'],
            attacks: List['Attack'],
            data_loader: DataLoader,
            iterations=10,
            save_results=False,
            print_results=True,
            device='gpu') -> dict:

        # count the number of misclassified images for each attack and each model
        # 1 - misclassified, 0 - classified correctly
        # transferability = {
        #     "att_name": {
        #         "model_name": [0,0,1,0,1,1,0,1]
        #     }
        # }

        if len(attacks) == 0:
            raise ValueError("No attacks provided")

        if len(trans_models) == 0:
            raise ValueError("No transferability models provided")

        transferability = {}
        it = 0

        pbar = tqdm(total=len(attacks) * iterations * len(trans_models))
        attacked_model.to(device)
        for trans_model in trans_models:
            trans_model.to(device)

        for images, labels in data_loader:
            if it >= iterations:
                break
            images, labels = images.to(device), labels.to(device)
            images, labels = ModelUtils.remove_missclassified(attacked_model, images, labels, device)
            if labels.numel() == 0:
                continue

            it += 1
            for attack in attacks:
                adv_images = attack(images, labels)
                for model in trans_models:
                    pbar.update(1)
                    model_name = model.__class__.__name__
                    attack_name = attack.attack
                    if attack_name not in transferability:
                        transferability[attack_name] = {}
                    if model_name not in transferability[attack_name]:
                        transferability[attack_name][model_name] = []

                    output = model(images)
                    _, prediction = torch.max(output, 1)

                    adv_output = model(adv_images)
                    _, adv_prediction = torch.max(adv_output, 1)
                    matches = torch.where(adv_prediction == prediction, torch.tensor(0), torch.tensor(1)).tolist()
                    transferability[attack_name][model_name].extend(matches)

        # sum the results for each attack and each model and present it as a percentage
        for attack_name, models in transferability.items():
            for model_name, results in models.items():
                transferability[attack_name][model_name] = sum(results) / len(results)

        # make 2d array of results. Each row is a attack, each column is a model
        results = []
        model_names = list(transferability[attack_name].keys())
        headers = ["Attacks"] + model_names
        results.append(headers)
        for attack_name, models in transferability.items():
            results.append([attack_name])
            for model_name, result in models.items():
                results[-1].append(result)

        if print_results:
            print()
            for result in results:
                line = ""
                for item in result:
                    line += f'{str(item):15s}'
                print(line)

        if save_results:
            folder_path = f"./results/transferability"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            with open(f'{folder_path}/{attacked_model.__class__.__name__}.txt', 'w') as file:
                for result in results:
                    line = ""
                    for item in result:
                        line += f'{str(item):15s}'
                    file.write(line + '\n')

        return transferability


    @staticmethod
    def transferability_attack_to_model(
            attacked_model_name: str,
            trans_models_names: List['torch.nn.Module'],
            attack_names: List['str'],
            save_path_folder=None,
            print_results=True,
            device='gpu') -> dict:

        # count the number of misclassified images for each attack and each model
        # 1 - misclassified, 0 - classified correctly
        # transferability = {
        #     "att_name": {
        #         "model_name": [0,0,1,0,1,1,0,1]
        #     }
        # }

        if len(trans_models_names) == 0:
            raise ValueError("No transferability models provided")

        name_to_model = {}
        transferability = {}
        imagenette_to_imagenet_index_map = ImageNetteClasses.get_imagenette_to_imagenet_map_by_index()

        for model_name in trans_models_names:
            name_to_model[model_name] = ImageNetModels.get_model(model_name=model_name)

        model_name_pbar = tqdm(total=len(attack_names) * (len(trans_models_names) - 1))
        for attack_name in attack_names:

            for model_name in trans_models_names:
                if model_name == attacked_model_name:
                    continue

                model_name_pbar.set_description(f"Model: {attacked_model_name:12s}|  Attack: {attack_name:10s}|  Trasferability Model: {model_name:12s}|")
                model_name_pbar.update(1)
                try:
                    _, test_loader = DatasetLoader.get_attacked_imagenette_dataset(attacked_model_name, attack_name, batch_size=4)
                except Exception as e:
                    print(f"Failed to load attacked dataset for {model_name} and {attack_name}")
                    print(f"Error: {e}")
                    continue
                model = name_to_model[model_name]
                model.to(device)
                model.eval()

                for images, labels in test_loader:
                    for original, mapped in imagenette_to_imagenet_index_map.items():
                        mask = labels == original
                        labels[mask] = mapped

                    images, labels = images.to(device), labels.to(device)

                    if attack_name not in transferability:
                        transferability[attack_name] = {}
                    if model_name not in transferability[attack_name]:
                        transferability[attack_name][model_name] = []

                    output = model(images)
                    _, predictions = torch.max(output, 1)
                    matches = torch.where(labels == predictions, torch.tensor(0), torch.tensor(1)).tolist()
                    transferability[attack_name][model_name].extend(matches)
                    del images, labels, output, predictions, matches

                del model, test_loader
                torch.cuda.empty_cache()

        # sum the results for each attack and each model and present it as a percentage
        for attack_name, models in transferability.items():
            for model_name, results in models.items():
                transferability[attack_name][model_name] = round(sum(results) / len(results), 2)

        # make 2d array of results. Each row is a attack, each column is a model
        results = []
        model_names = list(transferability[attack_name].keys())
        headers = ["Attacks"] + model_names
        results.append(headers)
        for attack_name, models in transferability.items():
            results.append([attack_name])
            for model_name, result in models.items():
                results[-1].append(result)

        if print_results:
            print()
            for result in results:
                line = ""
                for item in result:
                    line += f'{str(item):15s}'
                print(line)

        if save_path_folder is not None:
            if not os.path.exists(save_path_folder):
                os.makedirs(save_path_folder)
            with open(f'{save_path_folder}/{attacked_model_name}.txt', 'w') as file:
                for result in results:
                    line = ""
                    for item in result:
                        line += f'{str(item):15s}'
                    file.write(line + '\n')

        return transferability
    
    @staticmethod
    def transferability_model_to_model(
            model_names: List['str'],
            attack_names: List['str'],
            save_path_folder=None,
            print_results=True,
            device='gpu') -> dict:

        # count the number of misclassified images for each attack and each model
        # 1 - misclassified, 0 - classified correctly
        # transferability = {
        #     "model_name": {
        #         "model_name": [0,0,1,0,1,1,0,1]
        #     }
        # }

        if len(model_names) == 0:
            raise ValueError("No transferability models provided")

        name_to_model = {}
        transferability = {}
        imagenette_to_imagenet_index_map = ImageNetteClasses.get_imagenette_to_imagenet_map_by_index()

        for model_name in model_names:
            name_to_model[model_name] = ImageNetModels.get_model(model_name=model_name)

        for ground_model_name in model_names:
            print(f"Ground model: {ground_model_name} Time: {datetime.now().strftime('%H:%M:%S')}")
            model_name_pbar = tqdm(total=len(attack_names) * len(model_names))
            for attack_name in attack_names:
                for model_name in model_names:
                    model_name_pbar.set_description(f"Ground model: {ground_model_name:12s}|  Attack: {attack_name:10s}|  Trasferability Model: {model_name:12s}|")
                    model_name_pbar.update(1)
                    try:
                        _, test_loader = DatasetLoader.get_attacked_imagenette_dataset(ground_model_name, attack_name)
                    except Exception as e:
                        print(f"Failed to load attacked dataset for {model_name} and {attack_name}")
                        print(f"Error: {e}")
                        continue

                    model = name_to_model[model_name]
                    model.to(device)
                    model.eval()

                    for images, labels in test_loader:
                        for original, mapped in imagenette_to_imagenet_index_map.items():
                            mask = labels == original
                            labels[mask] = mapped

                        images, labels = images.to(device), labels.to(device)

                        if ground_model_name not in transferability:
                            transferability[ground_model_name] = {}
                        if model_name not in transferability[ground_model_name]:
                            transferability[ground_model_name][model_name] = []

                        output = model(images)
                        _, predictions = torch.max(output, 1)
                        matches = torch.where(labels == predictions, torch.tensor(0), torch.tensor(1)).tolist()
                        transferability[ground_model_name][model_name].extend(matches)
                        del images, labels, output, predictions, matches

                    del model, test_loader
                    torch.cuda.empty_cache()

        # sum the results for each attack and each model and present it as a percentage
        for ground_model, models in transferability.items():
            for model_name, results in models.items():
                transferability[ground_model][model_name] = round(sum(results) / len(results), 2)

        # make 2d array of results. Each row is a attack, each column is a model
        results = []
        model_names = list(transferability[model_names[0]].keys())
        headers = ["Models"] + model_names
        results.append(headers)
        for ground_model, models in transferability.items():
            results.append([ground_model])
            for model_name, result in models.items():
                results[-1].append(result)

        if print_results:
            print()
            for result in results:
                line = ""
                for item in result:
                    line += f'{str(item):15s}'
                print(line)

        if save_path_folder is not None:
            if not os.path.exists(save_path_folder):
                os.makedirs(save_path_folder)
            with open(f'{save_path_folder}/m2mTransferability.txt', 'w') as file:
                for result in results:
                    line = ""
                    for item in result: 
                        line += f'{str(item):15s}'
                    file.write(line + '\n')

        return transferability