import torch
from typing import List


class AttackResult():
    def __init__(self, actual, predicted, adv_example):
        self.actual = actual
        self.predicted = predicted
        self.adv = adv_example

    def __str__(self) -> str:
        return f"Actual: {self.actual}, Predicted: {self.predicted}"


class Attack():
    def __init__(self, name: str, model: torch.nn.Module, num_of_adv_examples: int = 0):

        self.attack_name = name
        self.model = model
        self.num_of_adv_examples = num_of_adv_examples

        self.device = next(model.parameters()).device

    def attack(self, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, *args, **kwargs) -> List[AttackResult]:
        raise NotImplementedError
