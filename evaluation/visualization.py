from typing import List
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import gc

from domain.attack.attack_eval_score import AttackEvaluationScore
from domain.attack.attack_result import AttackResult
from domain.attack.multiattack_result import MultiattackResult


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
def plot_multiattacked_images(multiattack_results: MultiattackResult, classes_names: list, rgb=True, save_path_folder=None, visualize=True):
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
        diff_img = torch.where(diff_img != 0, torch.tensor(1.0), torch.tensor(0.0))
        diff_img = (diff_img.permute(1, 2, 0) if rgb else diff_img[0])
        plot_image(ax, diff_img, cmap=None if rgb else 'gray')

        ax = axes[2, img_cnt]
        adv_img = (attack_data.adv_image.permute(1, 2, 0) if rgb else attack_data.adv_image[0]).detach().cpu()
        plot_image(ax, adv_img, cmap=None if rgb else 'gray')

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

    torch.cuda.empty_cache()


@torch.no_grad()
def attack_visualization(source_image: torch.Tensor, attacked_image: torch.Tensor, 
                        model_name: str, attack_name: str, true_label: int = None, 
                        predicted_label: int = None, distance_score=None):
    """
    Visualize a single attack example showing original, adversarial, and difference images.
    
    Args:
        source_image: Original image tensor (CHW format)
        attacked_image: Adversarial image tensor (CHW format)
        model_name: Name of the model
        attack_name: Name of the attack
        true_label: True label of the image (optional)
        predicted_label: Predicted label after attack (optional)
        distance_score: AttackDistanceScore object with distance metrics (optional)
    """
    # Convert to CPU and display format (HWC)
    source_display = source_image.cpu().permute(1, 2, 0)
    attacked_display = attacked_image.cpu().permute(1, 2, 0)
    
    # Calculate the actual difference (adversarial - original)
    actual_diff = attacked_display - source_display
    
    # Create a colored difference map (red for positive, blue for negative changes)
    diff_colored = torch.zeros_like(source_display)
    
    # Process each RGB channel separately
    for channel in range(3):  # RGB channels
        channel_diff = actual_diff[:, :, channel]
        
        # Red channel: positive differences (adversarial > original)
        diff_colored[:, :, 0] += torch.clamp(channel_diff * 8, 0, 1)
        
        # Green channel: negative differences (adversarial < original)  
        diff_colored[:, :, 1] += torch.clamp(-channel_diff * 8, 0, 1)
    
    # Normalize to [0, 1] range
    diff_colored = torch.clamp(diff_colored, 0, 1)
    
    # Create the plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(source_display.clamp(0, 1))
    title_orig = 'Original'
    if true_label is not None:
        title_orig += f'\n(True: {true_label})'
    axes[0].set_title(title_orig)
    axes[0].axis('off')
    
    # Adversarial image
    axes[1].imshow(attacked_display.clamp(0, 1))
    title_adv = 'Adversarial'
    if predicted_label is not None:
        title_adv += f'\n(Pred: {predicted_label})'
    axes[1].set_title(title_adv)
    axes[1].axis('off')
    
    # Difference image
    axes[2].imshow(diff_colored)
    axes[2].set_title('Difference (Red: +, Green: -)')
    axes[2].axis('off')
    
    # Add main title below the images
    main_title = f'Model: {model_name} | Attack: {attack_name}'
    plt.figtext(0.5, 0.05, main_title, ha='center', fontsize=14, weight='bold')
    
    # Add distance metrics if available
    if distance_score is not None:
        l0 = distance_score.l0_pixels
        l1 = distance_score.l1
        l2 = distance_score.l2
        linf = distance_score.linf
        power = distance_score.power_mse
        
        distance_text = f'L0: {l0:.2f} | L1: {l1:.3f} | L2: {l2:.4f} | L∞: {linf:.4f} | Power: {power:.4f}'
        plt.figtext(0.5, 0.02, distance_text, ha='center', fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()