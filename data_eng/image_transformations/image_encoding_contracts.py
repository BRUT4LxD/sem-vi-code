from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict

import torch


class ImageEncodingStrategy(str, Enum):
    PERMUTATION = "permutation"
    ARNOLD_CAT_MAP = "arnold_cat_map"
    GAUSSIAN_BLUR = "gaussian_blur"
    MEDIAN_FILTER = "median_filter"
    BILATERAL_FILTER = "bilateral_filter"
    AVERAGE_SMOOTHING = "average_smoothing"


@dataclass(frozen=True)
class ImageEncodingParams:
    strategy: ImageEncodingStrategy
    seed: int = 0
    kernel_size: int = 5
    sigma: float = 1.0
    bilateral_color_sigma: float = 0.1
    bilateral_spatial_sigma: float = 1.0
    iterations: int = 1

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["strategy"] = self.strategy.value
        return data

    @staticmethod
    def from_dict(data: Dict) -> "ImageEncodingParams":
        return ImageEncodingParams(
            strategy=ImageEncodingStrategy(str(data["strategy"])),
            seed=int(data.get("seed", 0)),
            kernel_size=int(data.get("kernel_size", 5)),
            sigma=float(data.get("sigma", 1.0)),
            bilateral_color_sigma=float(data.get("bilateral_color_sigma", 0.1)),
            bilateral_spatial_sigma=float(data.get("bilateral_spatial_sigma", 1.0)),
            iterations=int(data.get("iterations", 1)),
        )


class ImageEncoder:
    def __init__(self, params: ImageEncodingParams):
        self.params = params

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, encoded_image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

