from typing import List, Dict, Optional
import torch
from tqdm import tqdm
import numpy as np
from domain.attack.attack_distance_score import AttackDistanceScore
from domain.attack.attack_eval_score import AttackEvaluationScore
from domain.attack.attack_result import AttackResult
from torchmetrics import Accuracy, Precision, Recall, F1Score

class Metrics:

    @staticmethod
    @torch.no_grad()
    def l0_elements(image1: torch.Tensor, image2: torch.Tensor, threshold: float = 1e-6):
        """
        Calculate L0 distance (number of non-zero elements in perturbation).
        Counts individual tensor elements (per-channel).
        
        Args:
            image1: Original image tensor
            image2: Adversarial image tensor
            threshold: Minimum change to consider an element as modified
            
        Returns:
            float: L0 distance (number of modified elements)
        """
        diff = torch.abs(image1 - image2)
        return (diff > threshold).sum().item()

    @staticmethod
    @torch.no_grad()
    def l0_pixels(image1: torch.Tensor, image2: torch.Tensor, threshold: float = 1e-6):
        """
        Calculate L0 distance (number of modified pixels).
        Counts pixels where ANY channel changed (collapses channels).
        
        Args:
            image1: Original image tensor (C,H,W) or (N,C,H,W)
            image2: Adversarial image tensor (C,H,W) or (N,C,H,W)
            threshold: Minimum change to consider a pixel as modified
            
        Returns:
            float or list: L0 distance (number of modified pixels)
        """
        diff = torch.abs(image1 - image2)
        if diff.ndim == 3:  # (C,H,W)
            changed = (diff > threshold).any(dim=0)  # collapse channels
            return changed.sum().item()
        elif diff.ndim == 4:  # (N,C,H,W)
            changed = (diff > threshold).any(dim=1)  # (N,H,W)
            return changed.view(diff.size(0), -1).sum(dim=1).tolist()
        else:
            raise ValueError("Expected (C,H,W) or (N,C,H,W) tensor")

    @staticmethod
    @torch.no_grad()
    def l1_distance(image1: torch.Tensor, image2: torch.Tensor):
        """
        Calculate L1 distance (Manhattan distance) using PyTorch.
        
        Args:
            image1: Original image tensor (expected in [0,1] range)
            image2: Adversarial image tensor (expected in [0,1] range)
            
        Returns:
            float: L1 distance (in [0,1] range)
        """
        return torch.norm(image1 - image2, p=1).item()

    @staticmethod
    @torch.no_grad()
    def l2_distance(image1: torch.Tensor, image2: torch.Tensor):
        """
        Calculate L2 distance (Euclidean distance) using PyTorch.
        
        Args:
            image1: Original image tensor (expected in [0,1] range)
            image2: Adversarial image tensor (expected in [0,1] range)
            
        Returns:
            float: L2 distance (in [0,1] range)
        """
        return torch.norm(image1 - image2, p=2).item()

    @staticmethod
    @torch.no_grad()
    def linf_distance(image1: torch.Tensor, image2: torch.Tensor):
        """
        Calculate L∞ distance (maximum absolute difference) using PyTorch.
        
        Args:
            image1: Original image tensor (expected in [0,1] range)
            image2: Adversarial image tensor (expected in [0,1] range)
            
        Returns:
            float: L∞ distance (in [0,1] range)
        """
        return torch.norm(image1 - image2, p=float('inf')).item()

    @staticmethod
    @torch.no_grad()
    def attack_power_mse(image1: torch.Tensor, image2: torch.Tensor):
        """
        Calculate attack power as mean squared perturbation per element.
        This follows signal/image processing convention for "power".
        
        Args:
            image1: Original image tensor
            image2: Adversarial image tensor
            
        Returns:
            float: Mean squared perturbation per element
        """
        diff = image1 - image2
        return diff.pow(2).mean().item()

    @staticmethod
    @torch.no_grad()
    def attack_power_mse_per_pixel(image1: torch.Tensor, image2: torch.Tensor):
        """
        Calculate attack power as mean squared perturbation per pixel (averaged over channels).
        
        Args:
            image1: Original image tensor (C,H,W)
            image2: Adversarial image tensor (C,H,W)
            
        Returns:
            float: Mean squared perturbation per pixel
        """
        diff = image1 - image2
        return diff.pow(2).mean(dim=0).mean().item()


    @staticmethod
    @torch.no_grad()
    def attack_distance_score(attack_results: List[AttackResult]) -> AttackDistanceScore:
        """
        Calculate average distance metrics across attack results.
        Uses L0 pixels and MSE power for standard conventions.
        """
        l0_pixels = 0.0
        l1 = 0.0
        l2 = 0.0
        lInf = 0.0
        power_mse = 0.0
        n = len(attack_results)

        if n == 0:
            return AttackDistanceScore(l0_pixels, l1, l2, lInf, power_mse)

        for result in attack_results:
            l0_pixels += Metrics.l0_pixels(result.src_image, result.adv_image)
            l1 += Metrics.l1_distance(result.src_image, result.adv_image)
            l2 += Metrics.l2_distance(result.src_image, result.adv_image)
            lInf += Metrics.linf_distance(result.src_image, result.adv_image)
            power_mse += Metrics.attack_power_mse(result.src_image, result.adv_image)

        l0_pixels = l0_pixels / n
        l1 = l1 / n
        l2 = l2 / n
        lInf = lInf / n
        power_mse = power_mse / n

        return AttackDistanceScore(l0_pixels, l1, l2, lInf, power_mse)

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
    def evaluate_attack(attack_results: List[AttackResult], num_classes: int, 
                       clean_accuracy: float = None) -> dict:
        """
        Evaluate comprehensive attack effectiveness with standard ASR definitions.
        
        Args:
            attack_results: List of attack results
            num_classes: Number of classes
            clean_accuracy: Model accuracy on clean images (0.0 to 1.0) - optional
            
        Returns:
            dict: Comprehensive attack effectiveness metrics (all in 0-1 range)
        """
        if len(attack_results) == 0:
            return {
                'adversarial_accuracy': clean_accuracy or 0.0,
                'asr_unconditional': 0.0,
                'asr_conditional': float('nan'),
                'accuracy_drop': None,
                'relative_accuracy_drop': None,
                'macro_precision_adv': 0.0,
                'macro_recall_adv': 0.0,
                'macro_f1_adv': 0.0,
                'l0_elements': 0.0,
                'l1': 0.0,
                'l2': 0.0,
                'linf': 0.0,
                'power_mse': 0.0
            }
        
        # Calculate adversarial accuracy (robust accuracy)
        adv_correct = sum(int(r.predicted == r.actual) for r in attack_results)
        adv_acc = adv_correct / len(attack_results)
        
        # Unconditional ASR: 1 - adversarial_accuracy
        asr_uncond = 1.0 - adv_acc
        
        # Conditional ASR: success conditioned on clean correctness
        # Check if AttackResult has clean prediction info
        has_clean_info = hasattr(attack_results[0], 'clean_pred') or hasattr(attack_results[0], 'was_correct_clean')
        
        if has_clean_info:
            # Count samples that were correct on clean
            cond_den = sum(int(
                getattr(r, 'was_correct_clean', 
                       getattr(r, 'clean_pred', None) == r.actual)
            ) for r in attack_results)
            
            if cond_den > 0:
                # Count successful attacks on originally correct samples
                cond_num = sum(int(
                    getattr(r, 'was_correct_clean', 
                           getattr(r, 'clean_pred', None) == r.actual) and
                    (r.predicted != r.actual)
                ) for r in attack_results)
                asr_cond = cond_num / cond_den
            else:
                asr_cond = float('nan')
        else:
            # Fallback: compute unconditional ASR (same as asr_uncond)
            # This is NOT conditional on clean correctness - just a fallback
            asr_cond = sum(int(r.predicted != r.actual) for r in attack_results) / len(attack_results)
        
        # Accuracy drop (only meaningful if clean_accuracy provided)
        if clean_accuracy is not None:
            acc_drop = max(0.0, clean_accuracy - adv_acc)
            rel_drop = acc_drop / clean_accuracy if clean_accuracy > 0 else 0.0
        else:
            acc_drop = None
            rel_drop = None
        
        # Calculate traditional attack metrics
        attack_eval = Metrics.evaluate_attack_score(attack_results, num_classes)
        
        # Distance metrics
        d = attack_eval.distance_score
        
        return {
            'adversarial_accuracy': adv_acc,
            'asr_unconditional': asr_uncond,
            'asr_conditional': asr_cond,
            'accuracy_drop': acc_drop,
            'relative_accuracy_drop': rel_drop,
            'macro_precision_adv': attack_eval.prec / 100.0,
            'macro_recall_adv': attack_eval.rec / 100.0,
            'macro_f1_adv': attack_eval.f1 / 100.0,
            'l0_pixels': d.l0_pixels,
            'l1': d.l1,
            'l2': d.l2,
            'linf': d.linf,
            'power_mse': d.power_mse
        }

    @staticmethod
    @torch.no_grad()
    def evaluate_model_torchmetrics(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, num_classes: int, verbose: bool = True):
        
        model.eval()
        device = next(model.parameters()).device
        
        accuracy = Accuracy(task='multiclass', num_classes=num_classes).to(device)
        precision = Precision(task='multiclass', num_classes=num_classes, average='macro').to(device)
        recall = Recall(task='multiclass', num_classes=num_classes, average='macro').to(device)
        f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro').to(device)
        
        for images, labels in tqdm(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            accuracy(preds, labels)
            precision(preds, labels)
            recall(preds, labels)
            f1(preds, labels)
        
        acc = accuracy.compute().item()
        prec = precision.compute().item()
        rec = recall.compute().item()
        f1_score = f1.compute().item()

        if verbose:
            print(f'Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1 Score: {f1_score}')
        
        return acc, prec, rec, f1_score

    @staticmethod
    def format_attack_metrics(metrics: Dict, precision: int = 3) -> str:
        """
        Format attack metrics for consistent reporting.
        
        Args:
            metrics: Dictionary from evaluate_attack()
            precision: Number of decimal places
            
        Returns:
            str: Formatted metrics string
        """
        lines = []
        lines.append("🎯 Attack Effectiveness Metrics:")
        lines.append(f"   Robust Accuracy: {metrics['adversarial_accuracy']:.{precision}f}")
        lines.append(f"   ASR (Unconditional): {metrics['asr_unconditional']:.{precision}f}")
        
        if not np.isnan(metrics['asr_conditional']):
            # Check if conditional ASR equals unconditional (fallback case)
            if abs(metrics['asr_conditional'] - metrics['asr_unconditional']) < 1e-6:
                lines.append(f"   ASR (Conditional): {metrics['asr_conditional']:.{precision}f} (fallback - no clean pred info)")
            else:
                lines.append(f"   ASR (Conditional): {metrics['asr_conditional']:.{precision}f}")
        else:
            lines.append(f"   ASR (Conditional): N/A (no clean prediction info)")
        
        if metrics['accuracy_drop'] is not None:
            lines.append(f"   Accuracy Drop: {metrics['accuracy_drop']:.{precision}f}")
            lines.append(f"   Relative Drop: {metrics['relative_accuracy_drop']:.{precision}f}")
        
        lines.append(f"   Macro Precision: {metrics['macro_precision_adv']:.{precision}f}")
        lines.append(f"   Macro Recall: {metrics['macro_recall_adv']:.{precision}f}")
        lines.append(f"   Macro F1: {metrics['macro_f1_adv']:.{precision}f}")
        
        lines.append("\n📏 Perturbation Metrics:")
        lines.append(f"   L0 Pixels: {metrics['l0_pixels']:.0f}")
        lines.append(f"   L1 Distance: {metrics['l1']:.{precision}f}")
        lines.append(f"   L2 Distance: {metrics['l2']:.{precision}f}")
        lines.append(f"   L∞ Distance: {metrics['linf']:.{precision}f}")
        lines.append(f"   Power (MSE): {metrics['power_mse']:.{precision}e}")
        
        return "\n".join(lines)
