import numpy as np

class AttackEvaluationScore():
    def __init__(self, acc: float, prec: float, rec: float, f1: float, conf_matrix: np.ndarray):
        self.acc = acc
        self.prec = prec
        self.rec = rec
        self.f1 = f1
        self.conf_matrix = conf_matrix

    def __str__(self) -> str:
        return f"Accuracy: {self.acc}%, Precision: {self.prec}%, Recall: {self.rec}%, F1: {self.f1}%"