from typing import List
import torch
import matplotlib.pyplot as plt
import numpy as np

from domain.attack_eval_score import AttackEvaluationScore
from domain.attack_result import AttackResult
from domain.multiattack_result import MultiattackResult


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
def plot_multiattacked_images(multiattack_results: MultiattackResult, classes_names: List[str], rgb=True):
    attack_results = multiattack_results.attack_results
    eval_scores = multiattack_results.eval_scores

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
            nrows=3, ncols=num_of_columns, figsize=(15, 10))
        img_cnt = 0
        for j in range(len(attack_results[i])):
            if img_cnt >= num_of_columns:
                break
            res = attack_results[i][j]
            if res.actual == res.predicted:
                continue
            ax = axes[0][img_cnt]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                "{} -> {}".format(classes_names[res.actual], classes_names[res.predicted]))
            if rgb:
                img = res.src_image.permute(1, 2, 0).detach().cpu()
                ax.imshow(img, interpolation='none')
            else:
                img = res.src_image[0].detach().cpu()
                ax.imshow(img, cmap='gray', interpolation='none')

            ax = axes[1][img_cnt]
            ax.set_xticks([])
            ax.set_yticks([])
            if rgb:
                img = res.src_image.permute(1, 2, 0).detach().cpu(
                ) - res.adv_image.permute(1, 2, 0).detach().cpu()
                img = img.abs()
                img = torch.where(img == 0, torch.ones_like(img), img)
                ax.imshow(img, interpolation='none')
            else:
                img = res.adv_image[0].detach().cpu(
                ) - res.src_image[0].detach().cpu()
                img = img.abs()
                img = torch.where(img == 0, torch.ones_like(img), img)
                ax.imshow(img, cmap='gray', interpolation='none')

            ax = axes[2][img_cnt]
            ax.set_xticks([])
            ax.set_yticks([])
            if rgb:
                img = res.adv_image.permute(1, 2, 0).detach().cpu()
                ax.imshow(img, interpolation='none')
            else:
                img = res.adv_image[0].detach().cpu()
                ax.imshow(img, cmap='gray', interpolation='none')
            img_cnt += 1

        fig.suptitle(eval_scores[i], fontsize=14)
        plt.tight_layout()
        plt.show()
