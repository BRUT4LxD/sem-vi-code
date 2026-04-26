from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from data_eng.image_transformations.image_encoding_contracts import (
    ImageEncoder,
    ImageEncodingStrategy,
)


class SmoothingImageEncoder(ImageEncoder):
    """Apply one of the configured smoothing filters to a CHW tensor."""

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        self._validate_image(image)
        strategy = self.params.strategy
        if strategy == ImageEncodingStrategy.GAUSSIAN_BLUR:
            return self._gaussian_blur(image)
        if strategy == ImageEncodingStrategy.MEDIAN_FILTER:
            return self._median_filter(image)
        if strategy == ImageEncodingStrategy.BILATERAL_FILTER:
            return self._bilateral_filter(image)
        if strategy == ImageEncodingStrategy.AVERAGE_SMOOTHING:
            return self._average_smoothing(image)
        raise ValueError(f"Unsupported smoothing strategy: {strategy}")

    def decode(self, encoded_image: torch.Tensor) -> torch.Tensor:
        raise ValueError(
            f"{self.params.strategy.value} is lossy and cannot be decoded to the original image"
        )

    def _average_smoothing(self, image: torch.Tensor) -> torch.Tensor:
        k = self._kernel_size()
        batch = image.unsqueeze(0)
        smoothed = F.avg_pool2d(
            batch,
            kernel_size=k,
            stride=1,
            padding=k // 2,
            count_include_pad=False,
        )
        return smoothed.squeeze(0)

    def _gaussian_blur(self, image: torch.Tensor) -> torch.Tensor:
        k = self._kernel_size()
        sigma = self._positive_float(self.params.sigma, "sigma")
        coords = torch.arange(k, dtype=image.dtype, device=image.device) - (k // 2)
        kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel = kernel_2d.expand(image.shape[0], 1, k, k)
        return self._depthwise_filter(image, kernel, k)

    def _median_filter(self, image: torch.Tensor) -> torch.Tensor:
        k = self._kernel_size()
        padded = F.pad(image.unsqueeze(0), (k // 2, k // 2, k // 2, k // 2), mode="reflect")
        patches = padded.unfold(2, k, 1).unfold(3, k, 1)
        patches = patches.contiguous().view(1, image.shape[0], image.shape[1], image.shape[2], k * k)
        return patches.median(dim=-1).values.squeeze(0)

    def _bilateral_filter(self, image: torch.Tensor) -> torch.Tensor:
        k = self._kernel_size()
        color_sigma = self._positive_float(
            self.params.bilateral_color_sigma,
            "bilateral_color_sigma",
        )
        spatial_sigma = self._positive_float(
            self.params.bilateral_spatial_sigma,
            "bilateral_spatial_sigma",
        )
        radius = k // 2
        padded = F.pad(
            image.unsqueeze(0),
            (radius, radius, radius, radius),
            mode="reflect",
        )
        center = image.unsqueeze(0)
        weighted_sum = torch.zeros_like(center)
        weights_sum = torch.zeros(
            (1, 1, image.shape[1], image.shape[2]),
            dtype=image.dtype,
            device=image.device,
        )

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                neighbor = padded[
                    :,
                    :,
                    radius + dy : radius + dy + image.shape[1],
                    radius + dx : radius + dx + image.shape[2],
                ]
                spatial_distance_sq = float(dx * dx + dy * dy)
                spatial_weight = math.exp(
                    -spatial_distance_sq / (2 * spatial_sigma ** 2)
                )
                color_distance_sq = ((neighbor - center) ** 2).sum(dim=1, keepdim=True)
                color_weight = torch.exp(
                    -color_distance_sq / (2 * color_sigma ** 2)
                )
                weight = color_weight * spatial_weight
                weighted_sum = weighted_sum + neighbor * weight
                weights_sum = weights_sum + weight

        return (weighted_sum / weights_sum.clamp_min(1e-12)).squeeze(0)

    def _depthwise_filter(
        self,
        image: torch.Tensor,
        kernel: torch.Tensor,
        kernel_size: int,
    ) -> torch.Tensor:
        padded = F.pad(
            image.unsqueeze(0),
            (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2),
            mode="reflect",
        )
        return F.conv2d(padded, kernel, groups=image.shape[0]).squeeze(0)

    def _kernel_size(self) -> int:
        k = int(self.params.kernel_size)
        if k <= 0 or k % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {k}")
        return k

    @staticmethod
    def _positive_float(value: float, name: str) -> float:
        value = float(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return value

    @staticmethod
    def _validate_image(image: torch.Tensor) -> None:
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"image must be a torch.Tensor, got {type(image)}")
        if image.ndim != 3:
            raise ValueError(f"image must be CHW tensor, got shape {tuple(image.shape)}")

