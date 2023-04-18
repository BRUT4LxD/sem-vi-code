from typing import List
import torch


class AttackResult():
    def __init__(self, actual, predicted, adv_example):
        self.actual = actual
        self.predicted = predicted
        self.adv = adv_example

    def __str__(self) -> str:
        return f"Actual: {self.actual}, Predicted: {self.predicted}"

    @torch.no_grad()
    @staticmethod
    def create_from_adv_image(model: torch.nn.Module, adv_images: torch.Tensor, source_label: torch.Tensor) -> List['AttackResult']:
        if(adv_images.shape[0] != source_label.shape[0]):
            raise ValueError(
                f'advImages.shape {adv_images.shape} and labels.shape {source_label.shape} must be the same')

        attack_results: List[AttackResult] = []
        model.eval()
        outputs = model(adv_images)
        _, predicted_labels = torch.max(outputs.data, 1)

        for predicted_label, source_label in zip(predicted_labels, source_label):
            attack_results.append(AttackResult(
                source_label.item(), predicted_label.item(), adv_images))

        return attack_results
