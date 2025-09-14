import numpy as np

from domain.attack_distance_score import AttackDistanceScore


class AttackEvaluationScore():
    """
    A class to store evaluation scores for an attack.

    Attributes:
        acc (float): The accuracy score.
        prec (float): The precision score.
        rec (float): The recall score.
        f1 (float): The F1 score.
        conf_matrix (np.ndarray): The confusion matrix.
        attack_name (str): The name of the attack.
        time (float): The time taken for the attack.
        n_samples (int): The number of samples in the dataset.
    """

    def __init__(self, acc: float, prec: float, rec: float, f1: float, conf_matrix: np.ndarray, distance_score: AttackDistanceScore, attack_name: str = None, time: float = None, n_samples: int = None):
        """
        Initializes an AttackEvaluationScore object.

        Args:
            acc (float): The accuracy score.
            prec (float): The precision score.
            rec (float): The recall score.
            f1 (float): The F1 score.
            conf_matrix (np.ndarray): The confusion matrix.
            attack_name (str, optional): The name of the attack. Defaults to None.
            time (float, optional): The time taken for the attack. Defaults to None.
            n_samples (int, optional): The number of samples in the dataset. Defaults to None.
        """
        self.acc = round(acc, 2)
        self.prec = round(prec, 2)
        self.rec = round(rec, 2)
        self.f1 = round(f1, 2)
        self.conf_matrix = conf_matrix
        self.distance_score = distance_score
        self.attack_name = attack_name
        self.time = time
        self.n_samples = n_samples

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

        return f"{model_name:12s}{acc:13s}{prec:13s}{rec:13s}{f1:13s}{self.distance_score}{time:25s}{n_samples:13s}"
