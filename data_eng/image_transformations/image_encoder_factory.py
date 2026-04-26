from __future__ import annotations

from typing import Dict, Type

from data_eng.image_transformations.image_encoding_contracts import (
    ImageEncoder,
    ImageEncodingParams,
    ImageEncodingStrategy,
)
from data_eng.image_transformations.arnold_cat_map_image_encoder import (
    ArnoldCatMapImageEncoder,
)
from data_eng.image_transformations.permutation_image_encoder import (
    PermutationImageEncoder,
)
from data_eng.image_transformations.smoothing_image_encoder import SmoothingImageEncoder


_ENCODERS: Dict[ImageEncodingStrategy, Type[ImageEncoder]] = {
    ImageEncodingStrategy.PERMUTATION: PermutationImageEncoder,
    ImageEncodingStrategy.ARNOLD_CAT_MAP: ArnoldCatMapImageEncoder,
    ImageEncodingStrategy.GAUSSIAN_BLUR: SmoothingImageEncoder,
    ImageEncodingStrategy.MEDIAN_FILTER: SmoothingImageEncoder,
    ImageEncodingStrategy.BILATERAL_FILTER: SmoothingImageEncoder,
    ImageEncodingStrategy.AVERAGE_SMOOTHING: SmoothingImageEncoder,
}


def create_image_encoder(params: ImageEncodingParams) -> ImageEncoder:
    try:
        encoder_type = _ENCODERS[params.strategy]
    except KeyError as exc:
        raise ValueError(f"Unsupported image encoding strategy: {params.strategy}") from exc
    return encoder_type(params)


__all__ = [
    "ImageEncoder",
    "ImageEncodingParams",
    "ImageEncodingStrategy",
    "ArnoldCatMapImageEncoder",
    "PermutationImageEncoder",
    "SmoothingImageEncoder",
    "create_image_encoder",
]

