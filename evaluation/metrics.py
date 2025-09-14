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
    def evaluate_attack_score(attack_results: List[AttackResult], num_classes: int) -> AttackEvaluationScore:
        """
        Evaluate attack effectiveness by measuring model performance on adversarial examples.
        
        Args:
            attack_results: List of attack results
            num_classes: Number of classes
            
        Returns:
            AttackEvaluationScore: Attack evaluation metrics
        """
        if len(attack_results) == 0:
            return AttackEvaluationScore(0.0, 0.0, 0.0, 0.0, np.zeros((num_classes, num_classes), dtype=np.int32), AttackDistanceScore(0.0, 0.0, 0.0, 0.0, 0.0))

        # Extract true labels and adversarial predictions
        true_labels = [result.actual for result in attack_results]
        adv_predictions = [result.predicted for result in attack_results]

        # Create confusion matrix: true_labels vs adversarial_predictions
        conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
        for true_label, adv_pred in zip(true_labels, adv_predictions):
            conf_matrix[true_label][adv_pred] += 1

        # Calculate accuracy (how often model predicts correctly on adversarial examples)
        accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)

        # Calculate per-class precision, recall, F1
        precision = []
        recall = []
        f1_score = []
        
        for i in range(num_classes):
            # True Positives: correctly predicted as class i
            tp = conf_matrix[i][i]
            # False Positives: incorrectly predicted as class i
            fp = np.sum(conf_matrix[:, i]) - tp
            # False Negatives: class i incorrectly predicted as other classes
            fn = np.sum(conf_matrix[i, :]) - tp

            # Calculate metrics with zero division handling
            precision_i = tp / (tp + fp) if (tp + fp) != 0 else 0.0
            recall_i = tp / (tp + fn) if (tp + fn) != 0 else 0.0
            f1_score_i = 2 * precision_i * recall_i / (precision_i + recall_i) if (precision_i + recall_i) != 0 else 0.0

            precision.append(precision_i)
            recall.append(recall_i)
            f1_score.append(f1_score_i)

        # Convert to percentages
        acc = accuracy * 100
        prec = np.mean(np.array(precision)) * 100
        rec = np.mean(np.array(recall)) * 100
        f1 = np.mean(np.array(f1_score)) * 100
        
        # Calculate distance metrics
        distances = Metrics.attack_distance_score(attack_results)

        return AttackEvaluationScore(acc, prec, rec, f1, conf_matrix, distances)

    @staticmethod
    @torch.no_grad()
    def evaluate_attack(attack_results: List[AttackResult], clean_accuracy: float, 
                                    num_classes: int) -> dict:
        """
        Evaluate comprehensive attack effectiveness including Attack Success Rate (ASR).
        
        Args:
            attack_results: List of attack results
            clean_accuracy: Model accuracy on clean images (0.0 to 1.0)
            num_classes: Number of classes
            
        Returns:
            dict: Comprehensive attack effectiveness metrics
        """
        if len(attack_results) == 0:
            return {
                'attack_success_rate': 0.0,
                'accuracy_drop': 0.0,
                'relative_accuracy_drop': 0.0,
                'robustness_score': 1.0,
                'degradation_ratio': 0.0,
                'adversarial_accuracy': clean_accuracy,
                'clean_accuracy': clean_accuracy,
                'successful_attacks': 0,
                'total_attacks': 0
            }
        
        # Calculate adversarial accuracy from attack results
        correct_predictions = sum(1 for result in attack_results if result.actual == result.predicted)
        adversarial_accuracy = correct_predictions / len(attack_results)
        
        # Calculate Attack Success Rate (ASR)
        # ASR = (Clean_Accuracy - Adversarial_Accuracy) / Clean_Accuracy
        attack_success_rate = max(0.0, (clean_accuracy - adversarial_accuracy) / clean_accuracy) if clean_accuracy > 0 else 0.0
        
        # Calculate accuracy drop metrics
        accuracy_drop = max(0.0, clean_accuracy - adversarial_accuracy)
        relative_accuracy_drop = accuracy_drop / clean_accuracy if clean_accuracy > 0 else 0.0
        
        # Count successful attacks (where prediction changed from correct to incorrect)
        successful_attacks = sum(1 for result in attack_results if result.actual != result.predicted)
        
        # Calculate traditional attack metrics
        attack_eval = Metrics.evaluate_attack_score(attack_results, num_classes)
        
        # Combine all metrics
        return {
            'clean_accuracy': clean_accuracy,
            'adversarial_accuracy': adversarial_accuracy,
            'attack_success_rate': attack_success_rate,
            'accuracy_drop': accuracy_drop,
            'relative_accuracy_drop': relative_accuracy_drop,
            'robustness_score': 1.0 - attack_success_rate,  # Higher robustness = lower ASR
            'degradation_ratio': accuracy_drop / clean_accuracy if clean_accuracy > 0 else 0.0,
            'successful_attacks': successful_attacks,
            'total_attacks': len(attack_results),
            'attack_accuracy': attack_eval.acc / 100.0,  # Convert to 0-1 range
            'attack_precision': attack_eval.prec / 100.0,
            'attack_recall': attack_eval.rec / 100.0,
            'attack_f1': attack_eval.f1 / 100.0,
            'l0_distance': attack_eval.distance_score.l0,
            'l1_distance': attack_eval.distance_score.l1,
            'l2_distance': attack_eval.distance_score.l2,
            'linf_distance': attack_eval.distance_score.linf,
            'attack_power': attack_eval.distance_score.power
        }

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
