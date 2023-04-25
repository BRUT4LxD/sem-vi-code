from typing import List
import torch
import matplotlib.pyplot as plt
import numpy as np

from domain.attack_eval_score import AttackEvaluationScore
from domain.attack_result import AttackResult


@torch.no_grad()
def simple_visualize(test_model, test_loader, batch_size, classes, device='gpu'):
    batch_cnt = 0
    for images, labels in test_loader:
        if batch_cnt > 1:
            break
        batch_cnt += 1
        images = images.to(device)
        labels = labels.to(device)
        outputs = test_model(images)
        _, predictions = torch.max(outputs, 1)
        print('predictions: ', ' '.join('%5s' %
                                        classes[predictions[j]] for j in range(batch_size)))

        images = images.cpu().numpy()
        images = np.transpose(images, (0, 2, 3, 1))
        for i in range(batch_size):
            plt.figure()
            plt.imshow(images[i])
            plt.title(
                f'predicted: {classes[predictions[i]]}, label: {classes[labels[i]]}')
            plt.show()


@torch.no_grad()
def plot_adversarial_examples(examples, attack_params):
    cnt = 0
    plt.figure(figsize=(7, 10))
    for i in range(len(attack_params)):
        for j in range(max(len(examples[i]), 5)):
            cnt += 1
            plt.subplot(len(attack_params), len(examples[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("params: {}".format(attack_params[i]), fontsize=14)
            orig, adv, ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def plot_attacked_images(attack_results: List[List[AttackResult]], adv_images: torch.Tensor, eval_scores: List[AttackEvaluationScore]):
    num_of_attacks = len(attack_results)
    num_of_columns = 0
    for i in range(num_of_attacks):
        cols = 0
        for j in range(len(attack_results[i])):
            if attack_results[i][j].actual != attack_results[i][j].predicted:
                cols += 1

        num_of_columns = min(max(num_of_columns, cols), 5)

    for i in range(num_of_attacks):
        fig, axes = plt.subplots(
            nrows=1, ncols=num_of_columns, figsize=(15, 10))
        img_cnt = 0
        for j in range(len(attack_results[i])):
            if img_cnt >= num_of_columns:
                break
            res = attack_results[i][j]
            if res.actual == res.predicted:
                continue
            ax = axes[img_cnt]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("{} -> {}".format(res.actual, res.predicted))
            ax.imshow(adv_images[i][j][0].detach(
            ).cpu(), cmap="gray", interpolation='none')
            img_cnt += 1

        fig.suptitle(eval_scores[i], fontsize=14)
        plt.tight_layout()
        plt.show()
