import os
import time
from typing import List
from attacks.attack import Attack
from domain.attack_eval_score import AttackEvaluationScore
from domain.attack_result import AttackResult
from domain.model_config import ModelConfig
from domain.multiattack_result import MultiattackResult
from evaluation.metrics import calculate_attack_distance_score, evaluate_attack
import torch
from torch.nn import Module
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from shared.model_utils import get_model_num_classes


def single_attack(attack: Attack, test_loader: DataLoader, device='cuda', iterations: int = 1):
    model: Module = attack.model
    last_layer = list(model.children())[-1]
    last_layer_outputs = list(last_layer.parameters())[-1]
    num_classes = len(last_layer_outputs)
    it = 0
    for images, labels in test_loader:
        if it >= iterations:
            break
        images = images.to(device)
        labels = labels.to(device)
        start = time.time()
        adv_images = attack(images, labels)
        attack_res = AttackResult.create_from_adv_image(
            model, adv_images, images, labels, model.__call__.__name__, attack.attack)
        end = time.time()
        ev = evaluate_attack(attack_res, num_classes)
        print('{}: samples: {}, {} ({} ms)'.format(attack.attack, labels.shape[0], ev,
              int((end-start)*1000)))
        it += 1

        del adv_images, attack_res


def multiattack(attacks: List[Attack], test_loader: DataLoader, device='cuda', print_results=True, iterations=10, save_results=False) -> MultiattackResult:
    evaluation_scores: List['AttackEvaluationScore'] = []
    attack_results: List[List['AttackResult']] = []

    pbar = tqdm(total=len(attacks) * iterations)
    it = -1
    for attack in attacks:
        it += 1
        attack_results.append([])
        torch.cuda.empty_cache()
        pbar.set_description(f'({attack.model_name}) {attack.attack:12s}')
        num_classes = get_model_num_classes(attack.model)
        att_time = 0

        itx = 0
        try:
            for images, labels in test_loader:
                pbar.update(1)
                if itx >= iterations:
                    break
                itx += 1
                images, labels = images.to(device), labels.to(device)
                images, labels = _remove_missclassified(
                    attack.model, images, labels, device)
                if labels.numel() == 0:
                    continue

                start = time.time()
                adv_images = attack(images, labels)
                end = time.time()
                att_time += end-start
                attack_res = AttackResult.create_from_adv_image(
                    attack.model, adv_images, images, labels, attack.model_name, attack.attack)
                del adv_images, images, labels
                attack_results[it].extend(attack_res)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(
                    f'WARNING: ran out of memory, skipping attack for model: {attack.model_name} and attack: {attack.attack}')
                continue

        ev = evaluate_attack(attack_results[it], num_classes)
        ev.set_after_attack(attack.attack, att_time, len(attack_results[it]))
        evaluation_scores.append(ev)

    if print_results:
        print("\n\n************* RESULTS *************\n\n")
        print(f'SAMPLES: {ev.n_samples}\n')
        for ev in evaluation_scores:
            print(ev)

    if save_results:
        folder_path = f"./results/attacks"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(f'{folder_path}/{attack.model_name}.txt', 'w') as file:
            for eval_score in evaluation_scores:
                file.write(str(eval_score) + '\n')

    return MultiattackResult(evaluation_scores, attack_results)


def transferability_attack(
        attacked_model: torch.nn.Module,
        trans_models: List['torch.nn.Module'],
        attacks: List['Attack'],
        data_loader: DataLoader,
        iterations=10,
        save_results=False,
        print_results=True,
        device='gpu'):

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
        images, labels = _remove_missclassified(
            attacked_model, images, labels, device)
        if labels.numel() == 0:
            continue

        it += 1
        for attack in attacks:
            adv_images = attack(images, labels)
            # for i in range(n_images):
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


def _remove_missclassified(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor, device: str) -> torch.Tensor:
    outputs = model(images)
    _, predictions = torch.max(outputs, 1)
    images = images[predictions == labels].clone().to(device)
    labels = labels[predictions == labels].clone().to(device)
    return images, labels
