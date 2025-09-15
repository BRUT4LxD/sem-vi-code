import numpy as np
from typing import Optional
from dataclasses import dataclass, field

from domain.attack.attack_distance_score import AttackDistanceScore


@dataclass
class AttackEvaluationScore:
    """
    A comprehensive dataclass to store evaluation scores for an attack.
    Combines traditional attack evaluation with effectiveness metrics.

    Note: Core metrics (acc, prec, rec, f1) are stored as percentage values (0-100).
    Effectiveness metrics are stored as ratios (0-1).

    Attributes:
        acc: The accuracy score (robust accuracy) - stored as percentage (0-100).
        prec: The precision score - stored as percentage (0-100).
        rec: The recall score - stored as percentage (0-100).
        f1: The F1 score - stored as percentage (0-100).
        conf_matrix: The confusion matrix.
        distance_score: Distance metrics.
        attack_name: The name of the attack.
        time: The time taken for the attack.
        n_samples: The number of samples in the dataset.
        
        # Attack effectiveness metrics (optional) - stored as ratios (0-1)
        adversarial_accuracy: Robust accuracy on adversarial examples (0-1).
        asr_unconditional: Unconditional Attack Success Rate (0-1).
        asr_conditional: Conditional Attack Success Rate (0-1).
        accuracy_drop: Accuracy drop from clean to adversarial (0-1).
        relative_accuracy_drop: Relative accuracy drop (0-1).
        clean_accuracy: Clean accuracy for comparison (0-1).
    """
    
    # Core evaluation metrics
    acc: float
    prec: float
    rec: float
    f1: float
    conf_matrix: np.ndarray
    distance_score: AttackDistanceScore
    attack_name: str
    model_name: str

    # Optional metadata
    time: Optional[float] = None
    n_samples: Optional[int] = None
    
    # Optional effectiveness metrics
    adversarial_accuracy: Optional[float] = None
    asr_unconditional: Optional[float] = None
    asr_conditional: Optional[float] = None
    accuracy_drop: Optional[float] = None
    relative_accuracy_drop: Optional[float] = None
    clean_accuracy: Optional[float] = None

    def set_after_attack(self, time: int, n_samples: int):
        """
        Sets the attack name, time, and number of samples.

        Args:
            attack_name: The name of the attack.
            time: The time taken for the attack.
            n_samples: The number of samples in the dataset.
        """
        self.time = time
        self.n_samples = n_samples

    def set_effectiveness_metrics(self, adversarial_accuracy: float, asr_unconditional: float, 
                                 asr_conditional: Optional[float] = None, accuracy_drop: Optional[float] = None,
                                 relative_accuracy_drop: Optional[float] = None, clean_accuracy: Optional[float] = None):
        """
        Sets the attack effectiveness metrics.

        Args:
            adversarial_accuracy: Robust accuracy on adversarial examples (0-1).
            asr_unconditional: Unconditional Attack Success Rate (0-1).
            asr_conditional: Conditional Attack Success Rate (0-1).
            accuracy_drop: Accuracy drop from clean to adversarial (0-1).
            relative_accuracy_drop: Relative accuracy drop (0-1).
            clean_accuracy: Clean accuracy for comparison (0-1).
        """
        self.adversarial_accuracy = adversarial_accuracy
        self.asr_unconditional = asr_unconditional
        self.asr_conditional = asr_conditional
        self.accuracy_drop = accuracy_drop
        self.relative_accuracy_drop = relative_accuracy_drop
        self.clean_accuracy = clean_accuracy

    def has_effectiveness_metrics(self) -> bool:
        """
        Check if effectiveness metrics are available.

        Returns:
            True if effectiveness metrics are set, False otherwise.
        """
        return self.adversarial_accuracy is not None and self.asr_unconditional is not None

    def to_dict(self) -> dict:
        """
        Convert the dataclass to a dictionary for serialization.
        
        Note: Core metrics (acc, prec, rec, f1) are stored as percentages (0-100).
        Effectiveness metrics are stored as ratios (0-1).
        
        Returns:
            Dictionary representation of the evaluation score.
        """
        return {
            'acc': self.acc,
            'prec': self.prec,
            'rec': self.rec,
            'f1': self.f1,
            'conf_matrix': self.conf_matrix.tolist() if isinstance(self.conf_matrix, np.ndarray) else self.conf_matrix,
            'distance_score': {
                'l0_pixels': self.distance_score.l0_pixels,
                'l1': self.distance_score.l1,
                'l2': self.distance_score.l2,
                'linf': self.distance_score.linf,
                'power_mse': self.distance_score.power_mse
            },
            'attack_name': self.attack_name,
            'model_name': self.model_name,
            'time': self.time,
            'n_samples': self.n_samples,
            'adversarial_accuracy': self.adversarial_accuracy,
            'asr_unconditional': self.asr_unconditional,
            'asr_conditional': self.asr_conditional,
            'accuracy_drop': self.accuracy_drop,
            'relative_accuracy_drop': self.relative_accuracy_drop,
            'clean_accuracy': self.clean_accuracy
        }

    def __str__(self) -> str:
        """
        Returns a string representation of the evaluation scores.

        Returns:
            str: A string representation of the evaluation scores.
        """
        acc = f'acc: {self.acc}% '
        prec = f'prec: {self.prec}% '
        rec = f'rec: {self.rec}% '
        f1 = f'f1: {self.f1}% '
        if (self.time is not None) and (self.n_samples is not None) and (self.n_samples > 0):
            per_img_ms = 1000.0 * self.time / self.n_samples
            time = f'time: {per_img_ms:.2f}ms/img '
        else:
            time = ""
        n_samples = f'n_samples: {self.n_samples}' if self.n_samples is not None else ""

        # Basic metrics line
        basic_line = f"{self.model_name:12s}{self.attack_name:12s}{acc:13s}{prec:13s}{rec:13s}{f1:13s}{self.distance_score}{time:25s}{n_samples:13s}"
        
        # Add effectiveness metrics if available
        if self.has_effectiveness_metrics():
            effectiveness_line = f"\n   Effectiveness: ASR_uncond: {self.asr_unconditional:.3f}"
            if self.asr_conditional is not None:
                effectiveness_line += f", ASR_cond: {self.asr_conditional:.3f}"
            if self.accuracy_drop is not None:
                effectiveness_line += f", Acc_drop: {self.accuracy_drop:.3f}"
            if self.clean_accuracy is not None:
                effectiveness_line += f", Clean_acc: {self.clean_accuracy:.3f}"
            return basic_line + effectiveness_line
        
        return basic_line
