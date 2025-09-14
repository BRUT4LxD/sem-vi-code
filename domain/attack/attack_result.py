from typing import List
import torch

from domain.attack.attack_distance_score import AttackDistanceScore


class AttackedImageResult():
    def __init__(self, adv_image: torch.Tensor, label: torch.Tensor, distance: AttackDistanceScore) -> None:
        self.adv_image = adv_image
        self.label = label
        self.distance = distance

class AttackResult():
    def __init__(self, actual, predicted, adv_image: torch.Tensor, src_image: torch.Tensor, model_name: str, attack_name: str):
        self.actual = actual
        self.predicted = predicted
        self.adv_image = adv_image
        self.src_image = src_image
        self.model_name = model_name
        self.attack_name = attack_name

    def __str__(self) -> str:
        return f"Actual: {self.actual}, Predicted: {self.predicted}"

    @torch.no_grad()
    @staticmethod
    def create_from_adv_image(
        model: torch.nn.Module,
        adv_images: torch.Tensor,
        src_images: torch.Tensor,
        source_label: torch.Tensor,
        model_name: str,
        attack_name: str,
        labels_mapper: dict = None) -> List['AttackResult']:
        if(adv_images.shape[0] != source_label.shape[0]):
            raise ValueError(
                f'advImages.shape {adv_images.shape} and labels.shape {source_label.shape} must be the same')

        attack_results: List[AttackResult] = []
        model.eval()
        outputs = model(adv_images)
        _, predicted_labels = torch.max(outputs.data, 1)

        if labels_mapper is not None:
            for original, mapped in labels_mapper.items():
                mask = predicted_labels == original
                predicted_labels[mask] = mapped

        for i, (predicted_label, source_label) in enumerate(zip(predicted_labels, source_label)):
            attack_results.append(AttackResult(
                source_label.item(), predicted_label.item(), adv_images[i], src_images[i], model_name, attack_name))

        return attack_results
