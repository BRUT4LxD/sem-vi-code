import datetime
from typing import List
from attacks.attack import Attack
from domain.attack_eval_score import AttackEvaluationScore
from domain.attack_result import AttackResult
from domain.multiattack_result import MultiattackResult
from evaluation.metrics import evaluate_attack
import torch
from torch.nn import Module
from torch.utils.data.dataloader import DataLoader
from tabulate import tabulate
from tqdm import tqdm


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
        start = datetime.datetime.now()
        adv_images = attack(images, labels)
        attack_res = AttackResult.create_from_adv_image(
            model, adv_images, labels)
        end = datetime.datetime.now()
        ev = evaluate_attack(attack_res, num_classes)
        print('{}: samples: {}, {} ({} ms)'.format(attack.attack, labels.shape[0], ev,
              int((end-start).total_seconds()*1000)))
        it += 1

        del adv_images, attack_res


def multiattack(attacks: List[Attack], test_loader: DataLoader, device='cuda', iterations: int = 1) -> MultiattackResult:
    evaluation_scores: List['AttackEvaluationScore'] = []
    attack_results: List['AttackResult'] = []
    adv_images_list: List[torch.Tensor] = []

    print("PROCESSING ATTACKS...")
    for attack in tqdm(attacks):
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
            start = datetime.datetime.now()
            adv_images = attack(images, labels)
            end = datetime.datetime.now()
            attack_res = AttackResult.create_from_adv_image(
                model, adv_images, labels)
            ev = evaluate_attack(attack_res, num_classes)
            ev.set_after_attack(attack.attack, int(
                (end-start).total_seconds()*1000), labels.shape[0])
            adv_images_list.append(adv_images)
            attack_results.append(attack_res)
            evaluation_scores.append(ev)
            it += 1

    print("\n\n************* RESULTS *************\n\n")
    for ev in evaluation_scores:
        print(ev)

    return MultiattackResult(evaluation_scores, attack_results, torch.stack(adv_images_list, dim=0))
