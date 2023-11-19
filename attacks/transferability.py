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
            images, labels = ModelUtils.remove_missclassified(
                attacked_model, images, labels, device)
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
                    matches = torch.where(adv_prediction == prediction, torch.tensor(
                        0), torch.tensor(1)).tolist()
                    transferability[attack_name][model_name].extend(matches)

        # sum the results for each attack and each model and present it as a percentage
        for attack_name, models in transferability.items():
            for model_name, results in models.items():
                transferability[attack_name][model_name] = sum(
                    results) / len(results)

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

    