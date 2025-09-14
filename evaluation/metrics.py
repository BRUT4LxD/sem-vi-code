from typing import List
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from domain.attack.attack_distance_score import AttackDistanceScore

from domain.attack.attack_eval_score import AttackEvaluationScore
from domain.attack.attack_result import AttackResult
from domain.model.model_config import ModelConfig

from torchmetrics import Accuracy, Precision, Recall, F1Score
from torchmetrics.functional import pairwise_manhattan_distance, pairwise_euclidean_distance

class Metrics:

    @staticmethod
    @torch.no_grad()
    def l0_distance(image1: torch.Tensor, image2: torch.Tensor, threshold: float = 1e-6):
        """
        Calculate L0 distance (number of non-zero elements in perturbation).
        
        Args:
            image1: Original image tensor
            image2: Adversarial image tensor
            threshold: Minimum change to consider a pixel as modified
            
        Returns:
            float: L0 distance (number of modified pixels)
        """
        diff = torch.abs(image1 - image2)
        return (diff > threshold).sum().item()

    @staticmethod
    @torch.no_grad()
    def l1_distance(image1: torch.Tensor, image2: torch.Tensor):
        """
        Calculate L1 distance (Manhattan distance) using PyTorch.
        
        Args:
            image1: Original image tensor
            image2: Adversarial image tensor
            
        Returns:
            float: L1 distance
        """
        return torch.norm(image1 - image2, p=1).item()

    @staticmethod
    @torch.no_grad()
    def l2_distance(image1: torch.Tensor, image2: torch.Tensor):
        """
        Calculate L2 distance (Euclidean distance) using PyTorch.
        
        Args:
            image1: Original image tensor
            image2: Adversarial image tensor
            
        Returns:
            float: L2 distance
        """
        return torch.norm(image1 - image2, p=2).item()

    @staticmethod
    @torch.no_grad()
    def linf_distance(image1: torch.Tensor, image2: torch.Tensor):
        """
        Calculate L∞ distance (maximum absolute difference) using PyTorch.
        
        Args:
            image1: Original image tensor
            image2: Adversarial image tensor
            
        Returns:
            float: L∞ distance
        """
        return torch.norm(image1 - image2, p=float('inf')).item()

    @staticmethod
    @torch.no_grad()
    def attack_power(image1: torch.Tensor, image2: torch.Tensor, threshold: float = 1e-6):
        """
        Calculate attack power as number of pixels that changed significantly.
        This is equivalent to L0 distance with a threshold.
        
        Args:
            image1: Original image tensor
            image2: Adversarial image tensor
            threshold: Minimum change to consider a pixel as modified
            
        Returns:
            float: Number of significantly changed pixels
        """
        return Metrics.l0_distance(image1, image2, threshold)


    @staticmethod
    @torch.no_grad()
    def attack_distance_score(attack_results: List[AttackResult]) -> AttackDistanceScore:
        l0 = 0.0
        l1 = 0.0
        l2 = 0.0
        lInf = 0.0
        power = 0.0
        n = len(attack_results)

        if n == 0:
            return AttackDistanceScore(l0, l1, l2, lInf, power)

        for result in attack_results:
            l0 += Metrics.l0_distance(result.src_image, result.adv_image)
            l1 += Metrics.l1_distance(result.src_image, result.adv_image)
            l2 += Metrics.l2_distance(result.src_image, result.adv_image)
            lInf += Metrics.linf_distance(result.src_image, result.adv_image)
            power += Metrics.attack_power(result.src_image, result.adv_image)

        l0 = l0 / n
        l1 = l1 / n
        l2 = l2 / n
        lInf = lInf / n
        power = power / n

        return AttackDistanceScore(l0, l1, l2, lInf, power)

    @staticmethod
    @torch.no_grad()
    def evaluate_attack(attack_results: List[AttackResult], num_classes: int) -> AttackEvaluationScore:
        if len(attack_results) == 0:
            return AttackEvaluationScore(0.0, 0.0, 0.0, 0.0, np.zeros((num_classes, num_classes), dtype=np.int32), AttackDistanceScore(0.0, 0.0, 0.0, 0.0, 0.0))

        actual = [result.actual for result in attack_results]
        predicted = [result.predicted for result in attack_results]

        conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

        for a, p in zip(actual, predicted):
            conf_matrix[a][p] += 1

        accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)

        precision = []
        recall = []
        f1_score = []
        for i in range(num_classes):
            tp = conf_matrix[i][i]
            fp = np.sum(conf_matrix[:, i]) - tp
            fn = np.sum(conf_matrix[i, :]) - tp

            precision_i = tp / (tp + fp) if (tp + fp) != 0 else 0.0
            recall_i = tp / (tp + fn) if (tp + fn) != 0 else 0.0
            f1_score_i = 2 * precision_i * recall_i / \
                (precision_i + recall_i) if (precision_i + recall_i) != 0 else 0.0

            precision.append(precision_i)
            recall.append(recall_i)
            f1_score.append(f1_score_i)

        acc = (accuracy * 100)
        prec = np.mean(np.array(precision) * 100)
        rec = np.mean(np.array(recall) * 100)
        f1 = np.mean(np.array(f1_score) * 100)
        distances = Metrics.attack_distance_score(attack_results)

        return AttackEvaluationScore(acc, prec, rec, f1, conf_matrix, distances)

    @staticmethod
    @torch.no_grad()
    def evaluate_model_torchmetrics(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, num_classes: int, verbose: bool = True):
        
        model.eval()
        device = next(model.parameters()).device
        
        # Initialize metrics
        accuracy = Accuracy(task='multiclass', num_classes=num_classes).to(device)
        precision = Precision(task='multiclass', num_classes=num_classes, average='macro').to(device)
        recall = Recall(task='multiclass', num_classes=num_classes, average='macro').to(device)
        f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro').to(device)
        
        for images, labels in tqdm(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # Update metrics
            accuracy(preds, labels)
            precision(preds, labels)
            recall(preds, labels)
            f1(preds, labels)
        
        # Compute final metrics
        acc = accuracy.compute().item()
        prec = precision.compute().item()
        rec = recall.compute().item()
        f1_score = f1.compute().item()

        if verbose:
            print(f'Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1 Score: {f1_score}')
        
        return acc, prec, rec, f1_score
