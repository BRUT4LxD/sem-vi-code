from enum import Enum
import torch


class YOLOv5Models(Enum):
    NANO = "yolov5n"
    SMALL = "yolov5s"
    MEDIUM = "yolov5m"
    LARGE = "yolov5l"
    XLARGE = "yolov5x"


def getYOLOTopKPredictions(predictions: torch.Tensor, k: int) -> torch.Tensor:
    """
    Returns the top k predictions from the predictions tensor
    """

    top_k_predictions = torch.topk(predictions, k=k, dim=2)
    return top_k_predictions
