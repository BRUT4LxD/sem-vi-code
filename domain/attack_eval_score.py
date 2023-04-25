import numpy as np


class AttackEvaluationScore():
    def __init__(self, acc: float, prec: float, rec: float, f1: float, conf_matrix: np.ndarray, attack_name: str = None, time: float = None, n_samples: int = None):
        self.acc = acc
        self.prec = prec
        self.rec = rec
        self.f1 = f1
        self.conf_matrix = conf_matrix
        self.attack_name = attack_name
        self.time = time
        self.n_samples = n_samples

    def set_after_attack(self, attack_name: str, time: int, n_samples: int):
        self.attack_name = attack_name
        self.time = time
        self.n_samples = n_samples

    def __str__(self) -> str:

        model_name = f'{self.attack_name}' if self.attack_name is not None else ""
        acc = f'acc: {self.acc}% '
        prec = f'prec: {self.prec}% '
        rec = f'rec: {self.rec}% '
        f1 = f'f1: {self.f1}% '
        time = f'time: {format(self.time*1000/self.n_samples, ".2f")}ms/img' if self.time is not None else ""

        return f"{model_name:12}{acc:12s}{prec:12s}{rec:12s}{f1:12s}{time:12s}"
