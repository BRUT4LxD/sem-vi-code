import time
from typing import List
from attacks.attack import Attack
from domain.attack_eval_score import AttackEvaluationScore
from domain.attack_result import AttackResult
from domain.multiattack_result import MultiattackResult
from evaluation.metrics import evaluate_attack
import torch
from torch.nn import Module
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from utils.model_utils import get_model_num_classes


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


def multiattack(attacks: List[Attack], test_loader: DataLoader, device='cuda', print_results=True) -> MultiattackResult:
    evaluation_scores: List['AttackEvaluationScore'] = []
    attack_results: List[List['AttackResult']] = []

    pbar = tqdm(attacks)
    it = -1
    for attack in pbar:
        it += 1
        attack_results.append([])
        torch.cuda.empty_cache()
        pbar.set_description(f'({attack.model_name}) {attack.attack:12s}')
        num_classes = get_model_num_classes(attack.model)
        att_time = 0

        for images, labels in test_loader:
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
            adv_images, images, labels = adv_images.detach(
            ).cpu(), images.detach().cpu(), labels.detach().cpu()
            attack_results[it].extend(attack_res)

        ev = evaluate_attack(attack_results[it], num_classes)
        print(f'time {att_time}')
        print(f'attack results len {len(attack_results[it])}')
        ev.set_after_attack(attack.attack, att_time, len(attack_results[it]))
        evaluation_scores.append(ev)

    if print_results:
        print("\n\n************* RESULTS *************\n\n")
        print(f'SAMPLES: {ev.n_samples}\n')
        for ev in evaluation_scores:
            print(ev)

    return MultiattackResult(evaluation_scores, attack_results)


def _remove_missclassified(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor, device: str) -> torch.Tensor:
    print(f'image shape: {images.shape} labels shape: {labels.shape}')
    outputs = model(images)
    _, predictions = torch.max(outputs, 1)
    print(f'labels: {labels} predictions: {predictions}')
    images = images[predictions == labels].to(device)
    labels = labels[predictions == labels].to(device)
    return images, labels
