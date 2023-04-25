import numpy as np


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

    def __init__(self, acc: float, prec: float, rec: float, f1: float, conf_matrix: np.ndarray, attack_name: str = None, time: float = None, n_samples: int = None):
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
        self.acc = acc
        self.prec = prec
        self.rec = rec
        self.f1 = f1
        self.conf_matrix = conf_matrix
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
        time = f'time: {format(self.time*1000/self.n_samples, ".2f")}ms/img' if self.time is not None else ""

        return f"{model_name:12}{acc:12s}{prec:12s}{rec:12s}{f1:12s}{time:12s}"
