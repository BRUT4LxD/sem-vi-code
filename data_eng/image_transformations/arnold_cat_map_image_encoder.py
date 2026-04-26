from __future__ import annotations

import torch
from data_eng.image_transformations.image_encoding_contracts import ImageEncoder


class ArnoldCatMapImageEncoder(ImageEncoder):
    """
    Reversible Arnold Cat Map scrambling for the centered square inside CHW tensors.

    For rectangular inputs, only the centered ``N x N`` square is scrambled,
    where ``N = min(H, W)``. Pixels outside that square are left untouched.

    For each pixel coordinate ``(x, y)`` in an N x N image:
        x' = (x + y) mod N
        y' = (x + 2y) mod N

    ``iterations`` controls how many times the map is applied.
    """

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        self._validate_image(image)
        encoded = image.clone()
        y_start, y_end, x_start, x_end = self._center_square_bounds(image)
        square = encoded[:, y_start:y_end, x_start:x_end]
        for _ in range(self._iterations()):
            square = self._encode_once(square)
        encoded[:, y_start:y_end, x_start:x_end] = square
        return encoded

    def decode(self, encoded_image: torch.Tensor) -> torch.Tensor:
        self._validate_image(encoded_image)
        decoded = encoded_image.clone()
        y_start, y_end, x_start, x_end = self._center_square_bounds(encoded_image)
        square = decoded[:, y_start:y_end, x_start:x_end]
        for _ in range(self._iterations()):
            square = self._decode_once(square)
        decoded[:, y_start:y_end, x_start:x_end] = square
        return decoded

    @staticmethod
    def _center_square_bounds(image: torch.Tensor) -> tuple[int, int, int, int]:
        _, h, w = image.shape
        n = min(h, w)
        y_start = (h - n) // 2
        x_start = (w - n) // 2
        return y_start, y_start + n, x_start, x_start + n

    def _encode_once(self, image: torch.Tensor) -> torch.Tensor:
        _, n, _ = image.shape
        y, x = self._coordinates(n, image.device)
        new_x = (x + y) % n
        new_y = (x + 2 * y) % n
        encoded = torch.empty_like(image)
        encoded[:, new_y, new_x] = image[:, y, x]
        return encoded

    def _decode_once(self, encoded_image: torch.Tensor) -> torch.Tensor:
        _, n, _ = encoded_image.shape
        y, x = self._coordinates(n, encoded_image.device)
        new_x = (x + y) % n
        new_y = (x + 2 * y) % n
        decoded = torch.empty_like(encoded_image)
        decoded[:, y, x] = encoded_image[:, new_y, new_x]
        return decoded

    @staticmethod
    def _coordinates(size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.arange(size, device=device)
        return torch.meshgrid(values, values, indexing="ij")

    def _iterations(self) -> int:
        iterations = int(self.params.iterations)
        if iterations <= 0:
            raise ValueError(f"iterations must be positive, got {iterations}")
        return iterations

    @staticmethod
    def _validate_image(image: torch.Tensor) -> None:
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"image must be a torch.Tensor, got {type(image)}")
        if image.ndim != 3:
            raise ValueError(f"image must be CHW tensor, got shape {tuple(image.shape)}")

