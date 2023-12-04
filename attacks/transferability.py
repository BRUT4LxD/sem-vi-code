import torch
import os
import csv
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import List
from typing import List
from attacks.attack import Attack
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from attacks.attack_names import AttackNames
from attacks.simple_attacks import SimpleAttacks
from config.imagenet_classes import ImageNetClasses
from config.imagenet_models import ImageNetModels
from config.imagenette_classes import ImageNetteClasses
from data_eng.dataset_loader import DatasetLoader
from datetime import datetime

from shared.model_utils import ModelUtils

class Transferability():
    
    @staticmethod
    def transferability_attack(
            model_names: List['str'],
            attack_names: List['str'],
            images_per_attack=20,
            attacking_batch_size=16,
            model_batch_size=2,
            save_folder_path=None,
            print_results=True,
            use_test_set=True,
            device='gpu') -> dict:

        # count the number of misclassified images for each attack and each model
        # 1 - misclassified, 0 - classified correctly
        # transferability = {
        #     "att_name": {
        #         "model_name": [0,0,1,0,1,1,0,1]
        #     }
        # }

        if len(attack_names) == 0:
            raise ValueError("No attacks provided")

        if len(model_names) == 0:
            raise ValueError("No transferability models provided")

        pbar = tqdm(total=len(attack_names) * len(model_names) * len(model_names))

        for ground_model_name in model_names:
            if save_folder_path is not None and os.path.exists(f'{save_folder_path}/{ground_model_name}.csv'):
                pbar.update(len(attack_names) * len(model_names))
                continue

            transferability = {}
            for attack_name in attack_names:
                b_size = attacking_batch_size
                while True:
                    try:
                        adv_image_results = SimpleAttacks.get_attacked_imagenette_images(attack_name, ground_model_name, images_per_attack, batch_size=b_size, use_test_set=use_test_set)
                        break
                    except Exception as e:
                        print(f"Failed to load attacked dataset for {ground_model_name} and {attack_name}")
                        b_size = b_size // 2
                        print(f'Lowering the batchsize by half. From {b_size * 2} to {b_size}')
                        continue
                adv_image_with_labels = [(res.adv_image, res.label) for res in adv_image_results]
                for model_name in model_names:
                    pbar.update(1)
                    pbar.set_description(f"Ground Model: {ground_model_name:20s} | Attack: {attack_name:10s} | Model: {model_name:20s}")
                    if attack_name not in transferability:
                        transferability[attack_name] = {}
                    if model_name not in transferability[attack_name]:
                        transferability[attack_name][model_name] = []

                    attacked_images_loader = DataLoader(adv_image_with_labels, batch_size=model_batch_size)
                    model = ImageNetModels.get_model(model_name)
                    model.to(device)
                    model.eval()

                    for images, labels in attacked_images_loader:
                        images, labels = images.to(device), labels.to(device)
                        output = model(images)
                        _, prediction = torch.max(output, 1)
                        matches = torch.where(prediction == labels, torch.tensor(0), torch.tensor(1)).tolist()
                        transferability[attack_name][model_name].extend(matches)

            for attack_name, models in transferability.items():
                for model_name, results in models.items():
                    transferability[attack_name][model_name] = -1 if len(results) == 0 else round(float(sum(results)) / len(results), 2)

            results = []
            model_names = list(transferability[attack_name].keys())
            headers = ["Attacks"] + model_names
            results.append(headers)
            for attack_name, models in transferability.items():
                row = [attack_name]
                row.extend(models.values())
                results.append(row)

            if print_results:
                print()
                for result_line in results:
                    line = ""
                    for item in result_line:
                        line += f'{str(item):20s}'
                    print(line)

            if save_folder_path is not None:
                if not os.path.exists(save_folder_path):
                    os.makedirs(save_folder_path)
                file_path = os.path.join(save_folder_path, f'{ground_model_name}.csv')
                with open(file_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    for result_line in results:
                        writer.writerow(result_line)

    @staticmethod
    def transferability_attack_to_model_from_images(
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
            images_per_attack=20,
            attacking_batch_size=16,
            model_batch_size=2,
            save_folder_path=None,
            print_results=True,
            use_test_set=True,
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

        transferability = {}
        pbar = tqdm(total=len(attack_names) * len(model_names) * len(model_names))

        for ground_model_name in model_names:
            transferability = {}
            for attack_name in attack_names:
                b_size = attacking_batch_size
                while True:
                    try:
                        adv_image_results = SimpleAttacks.get_attacked_imagenette_images(attack_name, ground_model_name, images_per_attack, batch_size=b_size, use_test_set=use_test_set)
                        break
                    except Exception as e:
                        print(f"Failed to load attacked dataset for {ground_model_name} and {attack_name}")
                        b_size = b_size // 2
                        print(f'Lowering the batchsize by half. From {b_size * 2} to {b_size}')
                        continue
                adv_image_with_labels = [(res.adv_image, res.label) for res in adv_image_results]
                for model_name in model_names:
                    pbar.update(1)
                    pbar.set_description(f"Ground Model: {ground_model_name:20s} | Attack: {attack_name:10s} | Model: {model_name:20s}")
                    if ground_model_name not in transferability:
                        transferability[ground_model_name] = {}
                    if model_name not in transferability[ground_model_name]:
                        transferability[ground_model_name][model_name] = []

                    attacked_images_loader = DataLoader(adv_image_with_labels, batch_size=model_batch_size)
                    model = ImageNetModels.get_model(model_name)
                    model.to(device)
                    model.eval()

                    for images, labels in attacked_images_loader:
                        images, labels = images.to(device), labels.to(device)
                        output = model(images)
                        _, prediction = torch.max(output, 1)
                        matches = torch.where(prediction == labels, torch.tensor(0), torch.tensor(1)).tolist()
                        transferability[ground_model_name][model_name].extend(matches)

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

        if save_folder_path is not None:
            if not os.path.exists(save_folder_path):
                os.makedirs(save_folder_path)
            file_path = os.path.join(save_folder_path, 'm2mtransfer.csv')
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                for result_line in results:
                    writer.writerow(result_line)

        return transferability

    @staticmethod
    def transferability_model_to_model_from_files(model_names: List['str'], source_folder: str, save_folder_path=None, print_results=True) -> dict:
        model_files = {}
        # load files from source folder using model names as filename
        for model_name in model_names:
            # load csv file
            file_path = os.path.join(source_folder, f'{model_name}.csv')
            if not os.path.exists(file_path):
                print(f'File {file_path} does not exist')
                continue
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                model_files[model_name] = list(reader)
        
        model_to_model_transferability = []
        headers = ["Models"]
        for model_name in model_names:
            headers.append(model_name)

        model_to_model_transferability.append(headers)
        i = 0
        for model_name in model_files.keys():
            # skip the first row as it contains the model names
            model_to_attack_trans = model_files[model_name][1:]
            sum_accuracies = [0.0 for _ in model_names]
            for line in model_to_attack_trans:
                line_without_attack = line[1:]
                for i in range(len(line_without_attack)):
                    sum_accuracies[i] += round(float(line_without_attack[i]), 2)

            average_accuracies = [round(accuracy / len(model_to_attack_trans), 2) for accuracy in sum_accuracies]
            trans_arr = []
            trans_arr.append(model_name)
            trans_arr.extend(average_accuracies)
            model_to_model_transferability.append(trans_arr)

        if print_results:
            print(model_to_model_transferability)

        if save_folder_path is not None:
            if not os.path.exists(save_folder_path):
                os.makedirs(save_folder_path)
            file_path = os.path.join(save_folder_path, 'm2mtransferability.csv')
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                for result_line in model_to_model_transferability:
                    writer.writerow(result_line)

        return model_to_model_transferability

    @staticmethod
    def transferability_attack_to_model_from_files_summary(model_names: List['str'], attack_names: List['str'], source_folder: str, save_folder_path=None, print_results=True) -> dict:

        model_files = {}
        for model_name in model_names:
            file_path = os.path.join(source_folder, f'{model_name}.csv')
            if not os.path.exists(file_path):
                print(f'File {file_path} does not exist')
                continue
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                model_files[model_name] = list(reader)

        attack_to_model_transferability = []
        headers = ["Models"]
        for model_name in model_names:
            headers.append(model_name)

        attack_to_model_transferability.append(headers)
        sum_accuracies = {}
        for attack_name in attack_names:
            sum_accuracies[attack_name] = [-1.0 for _ in model_names]

        for model_name in model_files.keys():
            model_to_attack_trans = model_files[model_name][1:]
            for line in model_to_attack_trans:
                att_name = line[0]
                line_without_attack = line[1:]
                for i in range(len(line_without_attack)):
                    sum_accuracies[att_name][i] += float(line_without_attack[i])

        for att in sum_accuracies.keys():
            for i in range(len(sum_accuracies[att])):
                sum_accuracies[att][i] = round(sum_accuracies[att][i] / (len(model_files.keys()) - 1), 2)

        for att in sum_accuracies.keys():
            trans_arr = []
            trans_arr.append(att)
            trans_arr.extend(sum_accuracies[att])
            attack_to_model_transferability.append(trans_arr)

        if print_results:
            print(attack_to_model_transferability)

        if save_folder_path is not None:
            if not os.path.exists(save_folder_path):
                os.makedirs(save_folder_path)
            file_path = os.path.join(save_folder_path, 'a2mtransferability.csv')
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                for result_line in attack_to_model_transferability:
                    writer.writerow(result_line)

        return attack_to_model_transferability
