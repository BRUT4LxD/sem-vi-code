from typing import List, Dict, Optional, TYPE_CHECKING
import torch
from tqdm import tqdm
import numpy as np
from domain.attack.attack_distance_score import AttackDistanceScore
from domain.attack.attack_eval_score import AttackEvaluationScore
from domain.attack.attack_result import AttackResult
from torchmetrics import Accuracy, Precision, Recall, F1Score

if TYPE_CHECKING:
    from domain.attack.attack_eval_score import AttackEvaluationScore

class Metrics:

    @staticmethod
    def denormalize_image(image: torch.Tensor, mean: tuple = (0.485, 0.456, 0.406), 
                         std: tuple = (0.229, 0.224, 0.225)) -> torch.Tensor:
        """
        Denormalize an image tensor from ImageNet normalization to [0,1] range.
        
        Args:
            image: Normalized image tensor (C,H,W) or (N,C,H,W)
            mean: Mean values used for normalization (default ImageNet)
            std: Standard deviation values used for normalization (default ImageNet)
            
        Returns:
            torch.Tensor: Denormalized image in [0,1] range
        """
        # Convert to tensor if needed
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)
        
        # Ensure mean and std are tensors with correct shape
        if image.ndim == 3:  # (C,H,W)
            mean = torch.tensor(mean).view(3, 1, 1)
            std = torch.tensor(std).view(3, 1, 1)
        elif image.ndim == 4:  # (N,C,H,W)
            mean = torch.tensor(mean).view(1, 3, 1, 1)
            std = torch.tensor(std).view(1, 3, 1, 1)
        else:
            raise ValueError("Expected (C,H,W) or (N,C,H,W) tensor")
        
        # Move to same device as image
        mean = mean.to(image.device)
        std = std.to(image.device)
        
        # Denormalize: x = (x_norm * std) + mean
        denormalized = image * std + mean
        
        # Clamp to [0,1] range
        return torch.clamp(denormalized, 0.0, 1.0)

    @staticmethod
    @torch.no_grad()
    def l0_elements(image1: torch.Tensor, image2: torch.Tensor, threshold: float = 1e-6):
        """
        Calculate L0 distance (number of non-zero elements in perturbation).
        Counts individual tensor elements (per-channel).
        
        Args:
            image1: Original image tensor (expected in [0,1] range)
            image2: Adversarial image tensor (expected in [0,1] range)
            threshold: Minimum change to consider an element as modified
            
        Returns:
            float: L0 distance (number of modified elements)
        """
        # Validate input range
        assert (0.0 <= image1).all() and (image1 <= 1.0).all(), \
            "l0_elements expects [0,1] range; denormalize first."
        assert (0.0 <= image2).all() and (image2 <= 1.0).all(), \
            "l0_elements expects [0,1] range; denormalize first."
        
        diff = torch.abs(image1 - image2)
        return (diff > threshold).sum().item()

    @staticmethod
    @torch.no_grad()
    def l0_pixels(image1: torch.Tensor, image2: torch.Tensor, threshold: float = 1e-6):
        """
        Calculate L0 distance (number of modified pixels).
        Counts pixels where ANY channel changed (collapses channels).
        
        Args:
            image1: Original image tensor (C,H,W) or (N,C,H,W) (expected in [0,1] range)
            image2: Adversarial image tensor (C,H,W) or (N,C,H,W) (expected in [0,1] range)
            threshold: Minimum change to consider a pixel as modified
            
        Returns:
            float or list: L0 distance (number of modified pixels)
        """
        # Validate input range
        assert (0.0 <= image1).all() and (image1 <= 1.0).all(), \
            "l0_pixels expects [0,1] range; denormalize first."
        assert (0.0 <= image2).all() and (image2 <= 1.0).all(), \
            "l0_pixels expects [0,1] range; denormalize first."
        
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
        # Validate input range
        assert (0.0 <= image1).all() and (image1 <= 1.0).all(), \
            "l1_distance expects [0,1] range; denormalize first."
        assert (0.0 <= image2).all() and (image2 <= 1.0).all(), \
            "l1_distance expects [0,1] range; denormalize first."
        
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
        # Validate input range
        assert (0.0 <= image1).all() and (image1 <= 1.0).all(), \
            "l2_distance expects [0,1] range; denormalize first."
        assert (0.0 <= image2).all() and (image2 <= 1.0).all(), \
            "l2_distance expects [0,1] range; denormalize first."
        
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
        # Validate input range
        assert (0.0 <= image1).all() and (image1 <= 1.0).all(), \
            "linf_distance expects [0,1] range; denormalize first."
        assert (0.0 <= image2).all() and (image2 <= 1.0).all(), \
            "linf_distance expects [0,1] range; denormalize first."
        
        return torch.norm(image1 - image2, p=float('inf')).item()

    @staticmethod
    @torch.no_grad()
    def attack_power_mse(image1: torch.Tensor, image2: torch.Tensor):
        """
        Calculate attack power as mean squared perturbation per element.
        This follows signal/image processing convention for "power".
        
        Args:
            image1: Original image tensor (expected in [0,1] range)
            image2: Adversarial image tensor (expected in [0,1] range)
            
        Returns:
            float: Mean squared perturbation per element
        """
        # Validate input range
        assert (0.0 <= image1).all() and (image1 <= 1.0).all(), \
            "attack_power_mse expects [0,1] range; denormalize first."
        assert (0.0 <= image2).all() and (image2 <= 1.0).all(), \
            "attack_power_mse expects [0,1] range; denormalize first."
        
        diff = image1 - image2
        return diff.pow(2).mean().item()

    @staticmethod
    @torch.no_grad()
    def attack_power_mse_per_pixel(image1: torch.Tensor, image2: torch.Tensor):
        """
        Calculate attack power as mean squared perturbation per pixel (averaged over channels).
        
        Args:
            image1: Original image tensor (C,H,W) (expected in [0,1] range)
            image2: Adversarial image tensor (C,H,W) (expected in [0,1] range)
            
        Returns:
            float: Mean squared perturbation per pixel
        """
        # Validate input range
        assert (0.0 <= image1).all() and (image1 <= 1.0).all(), \
            "attack_power_mse_per_pixel expects [0,1] range; denormalize first."
        assert (0.0 <= image2).all() and (image2 <= 1.0).all(), \
            "attack_power_mse_per_pixel expects [0,1] range; denormalize first."
        
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
    def evaluate_attack_score(attack_results: List[AttackResult], num_classes: int, 
                             clean_accuracy: Optional[float] = None) -> AttackEvaluationScore:
        """
        Evaluate attack effectiveness by measuring model performance on adversarial examples.
        
        Args:
            attack_results: List of attack results
            num_classes: Number of classes
            clean_accuracy: Optional clean accuracy for effectiveness metrics (0-1)
            
        Returns:
            AttackEvaluationScore: Comprehensive attack evaluation with optional effectiveness metrics.
            Core metrics (acc, prec, rec, f1) are stored as percentages (0-100).
            Effectiveness metrics are stored as ratios (0-1).
        """
        if len(attack_results) == 0:
            return AttackEvaluationScore(0.0, 0.0, 0.0, 0.0, 
            np.zeros((num_classes, num_classes), dtype=np.int32), 
            AttackDistanceScore(0.0, 0.0, 0.0, 0.0, 0.0), 
            "UNKNOWN_ATTACK", "UNKNOWN_MODEL")

        model_name = attack_results[0].model_name
        attack_name = attack_results[0].attack_name

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

        # Calculate effectiveness metrics if clean_accuracy is provided
        if clean_accuracy is not None:
            asr_unconditional = 1.0 - (acc / 100.0)  # Convert acc to 0-1 range and calculate ASR
            
            # Calculate conditional ASR if clean prediction info is available
            has_clean_info = hasattr(attack_results[0], 'clean_pred') or hasattr(attack_results[0], 'was_correct_clean')
            if has_clean_info:
                cond_den = sum(int(
                    getattr(r, 'was_correct_clean', 
                           getattr(r, 'clean_pred', None) == r.actual)
                ) for r in attack_results)
                
                if cond_den > 0:
                    cond_num = sum(int(
                        getattr(r, 'was_correct_clean', 
                               getattr(r, 'clean_pred', None) == r.actual) and
                        (r.predicted != r.actual)
                    ) for r in attack_results)
                    asr_conditional = cond_num / cond_den
                else:
                    asr_conditional = float('nan')
            else:
                asr_conditional = None
            
            accuracy_drop = max(0.0, clean_accuracy - (acc / 100.0))
            relative_accuracy_drop = accuracy_drop / clean_accuracy if clean_accuracy > 0 else 0.0
            
            return AttackEvaluationScore(
                acc, prec, rec, f1, conf_matrix, distances, attack_name, model_name,
                asr_unconditional=asr_unconditional,
                asr_conditional=asr_conditional,
                accuracy_drop=accuracy_drop,
                relative_accuracy_drop=relative_accuracy_drop,
                clean_accuracy=clean_accuracy
            )
        else:
            return AttackEvaluationScore(acc, prec, rec, f1, conf_matrix, distances, attack_name, model_name)


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
    def format_attack_metrics(score: AttackEvaluationScore, precision: int = 3) -> str:
        """
        Format attack metrics for consistent reporting.
        
        Args:
            score: AttackEvaluationScore from evaluate_attack_score()
            precision: Number of decimal places
            
        Returns:
            str: Formatted metrics string
        """
        lines = []
        lines.append("🎯 Attack Effectiveness Metrics:")
        
        if score.has_effectiveness_metrics():
            lines.append(f"   Robust Accuracy: {score.acc/100.0:.{precision}f}")
            lines.append(f"   ASR (Unconditional): {score.asr_unconditional:.{precision}f}")
            
            if score.asr_conditional is not None and not np.isnan(score.asr_conditional):
                # Check if conditional ASR equals unconditional (fallback case)
                if abs(score.asr_conditional - score.asr_unconditional) < 1e-6:
                    lines.append(f"   ASR (Conditional): {score.asr_conditional:.{precision}f} (fallback - no clean pred info)")
                else:
                    lines.append(f"   ASR (Conditional): {score.asr_conditional:.{precision}f}")
            else:
                lines.append(f"   ASR (Conditional): N/A (no clean prediction info)")
            
            if score.accuracy_drop is not None:
                lines.append(f"   Accuracy Drop: {score.accuracy_drop:.{precision}f}")
                lines.append(f"   Relative Drop: {score.relative_accuracy_drop:.{precision}f}")
            
            if score.clean_accuracy is not None:
                lines.append(f"   Clean Accuracy: {score.clean_accuracy:.{precision}f}")
        else:
            lines.append(f"   Accuracy: {score.acc:.{precision}f}%")
            lines.append(f"   Precision: {score.prec:.{precision}f}%")
            lines.append(f"   Recall: {score.rec:.{precision}f}%")
            lines.append(f"   F1-Score: {score.f1:.{precision}f}%")
        
        lines.append("\n📏 Perturbation Metrics:")
        lines.append(f"   L0 Pixels: {score.distance_score.l0_pixels:.0f}")
        lines.append(f"   L1 Distance: {score.distance_score.l1:.{precision}f}")
        lines.append(f"   L2 Distance: {score.distance_score.l2:.{precision}f}")
        lines.append(f"   L∞ Distance: {score.distance_score.linf:.{precision}f}")
        lines.append(f"   Power (MSE): {score.distance_score.power_mse:.{precision}e}")
        
        return "\n".join(lines)
