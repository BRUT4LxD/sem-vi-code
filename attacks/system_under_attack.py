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
            model, adv_images, labels)
        end = time.time()
        ev = evaluate_attack(attack_res, num_classes)
        print('{}: samples: {}, {} ({} ms)'.format(attack.attack, labels.shape[0], ev,
              int((end-start)*1000)))
        it += 1

        del adv_images, attack_res


def multiattack(attacks: List[Attack], test_loader: DataLoader, device='cuda') -> MultiattackResult:
    evaluation_scores: List['AttackEvaluationScore'] = []
    attack_results: List['AttackResult'] = []
    adv_images_list: List[torch.Tensor] = []

    print("PROCESSING ATTACKS...")
    for attack in tqdm(attacks):
        num_classes = get_model_num_classes(attack.model)
        images, labels = next(iter(test_loader))
        images = images.to(device)
        labels = labels.to(device)
        start = time.time()
        adv_images = attack(images, labels)
        end = time.time()
        attack_res = AttackResult.create_from_adv_image(
            attack.model, adv_images, labels)
        ev = evaluate_attack(attack_res, num_classes)
        ev.set_after_attack(attack.attack, end-start, labels.shape[0])
        adv_images_list.append(adv_images)
        attack_results.append(attack_res)
        evaluation_scores.append(ev)

    print("\n\n************* RESULTS *************\n\n")
    print(f'SAMPLES: {ev.n_samples}\n')
    for ev in evaluation_scores:
        print(ev)

    return MultiattackResult(evaluation_scores, attack_results, torch.stack(adv_images_list, dim=0))
