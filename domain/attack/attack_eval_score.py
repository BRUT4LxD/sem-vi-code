import numpy as np
from typing import Optional

from domain.attack.attack_distance_score import AttackDistanceScore


class AttackEvaluationScore():
    """
    A comprehensive class to store evaluation scores for an attack.
    Combines traditional attack evaluation with effectiveness metrics.

    Attributes:
        acc (float): The accuracy score (robust accuracy).
        prec (float): The precision score.
        rec (float): The recall score.
        f1 (float): The F1 score.
        conf_matrix (np.ndarray): The confusion matrix.
        distance_score (AttackDistanceScore): Distance metrics.
        attack_name (str): The name of the attack.
        time (float): The time taken for the attack.
        n_samples (int): The number of samples in the dataset.
        
        # Attack effectiveness metrics (optional)
        adversarial_accuracy (Optional[float]): Robust accuracy on adversarial examples.
        asr_unconditional (Optional[float]): Unconditional Attack Success Rate.
        asr_conditional (Optional[float]): Conditional Attack Success Rate.
        accuracy_drop (Optional[float]): Accuracy drop from clean to adversarial.
        relative_accuracy_drop (Optional[float]): Relative accuracy drop.
        clean_accuracy (Optional[float]): Clean accuracy for comparison.
    """

    def __init__(self, acc: float, prec: float, rec: float, f1: float, conf_matrix: np.ndarray, 
                 distance_score: AttackDistanceScore, attack_name: str = None, time: float = None, 
                 n_samples: int = None, adversarial_accuracy: Optional[float] = None,
                 asr_unconditional: Optional[float] = None, asr_conditional: Optional[float] = None,
                 accuracy_drop: Optional[float] = None, relative_accuracy_drop: Optional[float] = None,
                 clean_accuracy: Optional[float] = None):
        """
        Initializes an AttackEvaluationScore object.

        Args:
            acc (float): The accuracy score (robust accuracy).
            prec (float): The precision score.
            rec (float): The recall score.
            f1 (float): The F1 score.
            conf_matrix (np.ndarray): The confusion matrix.
            distance_score (AttackDistanceScore): Distance metrics.
            attack_name (str, optional): The name of the attack. Defaults to None.
            time (float, optional): The time taken for the attack. Defaults to None.
            n_samples (int, optional): The number of samples in the dataset. Defaults to None.
            adversarial_accuracy (Optional[float]): Robust accuracy on adversarial examples.
            asr_unconditional (Optional[float]): Unconditional Attack Success Rate.
            asr_conditional (Optional[float]): Conditional Attack Success Rate.
            accuracy_drop (Optional[float]): Accuracy drop from clean to adversarial.
            relative_accuracy_drop (Optional[float]): Relative accuracy drop.
            clean_accuracy (Optional[float]): Clean accuracy for comparison.
        """
        self.acc = acc
        self.prec = prec
        self.rec = rec
        self.f1 = f1
        self.conf_matrix = conf_matrix
        self.distance_score = distance_score
        self.attack_name = attack_name
        self.time = time
        self.n_samples = n_samples
        
        # Attack effectiveness metrics
        self.adversarial_accuracy = adversarial_accuracy
        self.asr_unconditional = asr_unconditional
        self.asr_conditional = asr_conditional
        self.accuracy_drop = accuracy_drop
        self.relative_accuracy_drop = relative_accuracy_drop
        self.clean_accuracy = clean_accuracy

    def set_after_attack(self, attack_name: str, time: int, n_samples: int):
        """
        Sets the attack name, time, and number of samples.

        Args:
            attack_name (str): The name of the attack.
            time (int): The time taken for the attack.
            n_samples (int): The number of samples in the dataset.
        """
        self.attack_name = attack_name
        self.time = time
        self.n_samples = n_samples

    def set_effectiveness_metrics(self, adversarial_accuracy: float, asr_unconditional: float, 
                                 asr_conditional: Optional[float] = None, accuracy_drop: Optional[float] = None,
                                 relative_accuracy_drop: Optional[float] = None, clean_accuracy: Optional[float] = None):
        """
        Sets the attack effectiveness metrics.

        Args:
            adversarial_accuracy (float): Robust accuracy on adversarial examples.
            asr_unconditional (float): Unconditional Attack Success Rate.
            asr_conditional (Optional[float]): Conditional Attack Success Rate.
            accuracy_drop (Optional[float]): Accuracy drop from clean to adversarial.
            relative_accuracy_drop (Optional[float]): Relative accuracy drop.
            clean_accuracy (Optional[float]): Clean accuracy for comparison.
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
            bool: True if effectiveness metrics are set, False otherwise.
        """
        return self.adversarial_accuracy is not None and self.asr_unconditional is not None

    def __str__(self) -> str:
        """
        Returns a string representation of the evaluation scores.

        Returns:
            str: A string representation of the evaluation scores.
        """
        model_name = f'{self.attack_name}' if self.attack_name is not None else ""
        acc = f'acc: {self.acc}% '
        prec = f'prec: {self.prec}% '
        rec = f'rec: {self.rec}% '
        f1 = f'f1: {self.f1}% '
        time = f'time: {format(self.time*1000/self.n_samples, ".2f")}ms/img ' if self.time and self.n_samples > 0 is not None else ""
        n_samples = f'n_samples: {self.n_samples}' if self.n_samples is not None else ""

        # Basic metrics line
        basic_line = f"{model_name:12s}{acc:13s}{prec:13s}{rec:13s}{f1:13s}{self.distance_score}{time:25s}{n_samples:13s}"
        
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
