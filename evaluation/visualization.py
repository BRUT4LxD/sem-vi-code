from typing import List
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import gc

from domain.attack_eval_score import AttackEvaluationScore
from domain.attack_result import AttackResult
from domain.multiattack_result import MultiattackResult


@torch.no_grad()
def simple_visualize(test_model, test_loader, batch_size, classes, device='cuda'):
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
def plot_multiattacked_images(multiattack_results, classes_names, rgb=True, save_path_folder=None, visualize=True):
    attack_results = multiattack_results.attack_results
    eval_scores = multiattack_results.eval_scores

    def plot_image(ax, img, cmap=None):
        """Helper function to plot an image on an axis."""
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(img, cmap=cmap, interpolation='none')
        del ax

    def process_images_and_plot(i, img_cnt, attack_data):
        """Handles the processing of images and plotting on axes."""
        ax = axes[0, img_cnt]
        ax.set_title(f"{classes_names[attack_data.actual]} -> {classes_names[attack_data.predicted]}")
        img = (attack_data.src_image.permute(1, 2, 0) if rgb else attack_data.src_image[0]).detach().cpu()
        plot_image(ax, img, cmap=None if rgb else 'gray')

        ax = axes[1, img_cnt]
        diff_img = torch.abs(attack_data.src_image - attack_data.adv_image).detach().cpu()
        diff_img = (diff_img.permute(1, 2, 0) if rgb else diff_img[0])
        plot_image(ax, diff_img, cmap=None if rgb else 'gray')

        ax = axes[2, img_cnt]
        adv_img = (attack_data.adv_image.permute(1, 2, 0) if rgb else attack_data.adv_image[0]).detach().cpu()
        plot_image(ax, adv_img, cmap=None if rgb else 'gray')

        del ax, img, diff_img, adv_img

    max_columns = min(max(len(a) for a in attack_results), 5)

    if max_columns == 0:
        return

    for i in range(len(attack_results)):
        fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 10))
        img_cnt = 0
        for attack_data in attack_results[i]:
            if img_cnt >= max_columns or attack_data.actual == attack_data.predicted:
                continue

            process_images_and_plot(i, img_cnt, attack_data)
            img_cnt += 1

            # Remove any reference to tensors to free up GPU memory
            del attack_data.src_image
            del attack_data.adv_image
            torch.cuda.empty_cache()

        fig.suptitle(eval_scores[i], fontsize=14)
        plt.tight_layout()

        if visualize:
            plt.show()

        if save_path_folder is not None and len(attack_results[i]) > 0:
            save_path = f".{save_path_folder}/{attack_results[i][0].model_name}"
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(f'{save_path}/{attack_results[i][0].model_name}_{eval_scores[i].attack_name}.png')

        # Cleanup to avoid potential memory leaks
        plt.close(fig)
        gc.collect()

    # Addtionally, cleanup the entire list of attack results
    del attack_results, eval_scores
    torch.cuda.empty_cache()
    gc.collect()

