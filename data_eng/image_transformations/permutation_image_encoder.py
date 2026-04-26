from __future__ import annotations

import torch

from data_eng.image_transformations.image_encoding_contracts import ImageEncoder


class PermutationImageEncoder(ImageEncoder):
    """
    Reversible spatial pixel permutation.

    Input is expected to be a CHW tensor. The same H*W permutation is applied to
    every channel, preserving RGB channel values at each permuted pixel.
    """

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        self._validate_image(image)
        c, h, w = image.shape
        permutation = self._permutation(h * w, image.device)
        flat = image.reshape(c, h * w)
        return flat[:, permutation].reshape_as(image)

    def decode(self, encoded_image: torch.Tensor) -> torch.Tensor:
        self._validate_image(encoded_image)
        c, h, w = encoded_image.shape
        permutation = self._permutation(h * w, encoded_image.device)
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(
            permutation.numel(),
            device=encoded_image.device,
            dtype=permutation.dtype,
        )
        flat = encoded_image.reshape(c, h * w)
        return flat[:, inverse].reshape_as(encoded_image)

    def _permutation(self, num_pixels: int, device: torch.device) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.params.seed)
        permutation = torch.randperm(num_pixels, generator=generator)
        return permutation.to(device)

    @staticmethod
    def _validate_image(image: torch.Tensor) -> None:
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"image must be a torch.Tensor, got {type(image)}")
        if image.ndim != 3:
            raise ValueError(f"image must be CHW tensor, got shape {tuple(image.shape)}")

