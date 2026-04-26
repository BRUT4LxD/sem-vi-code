from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm
from PIL import Image, PngImagePlugin

from data_eng.dataset_loader import load_imagenette
from data_eng.transforms import no_transformer
from data_eng.image_transformations import (
    ImageEncodingParams,
    ImageEncodingStrategy,
    create_image_encoder,
)


@dataclass(frozen=True)
class SavedEncodedImage:
    path: str
    label: int
    class_name: str
    source_path: str


@dataclass(frozen=True)
class ImageNetteImageTransformerConfig:
    output_root: str = "./data/imagenette_encoded"
    encoding: ImageEncodingParams = field(
        default_factory=lambda: ImageEncodingParams(
            strategy=ImageEncodingStrategy.PERMUTATION,
            seed=42,
        )
    )


class ImageNetteImageTransformer:
    """
    Encode ImageNette images and save each encoded image with decode metadata.

    Each output is a normal image file with the same name/extension as the
    source image. Encoding params and source metadata are embedded in image
    metadata where supported by Pillow (JPEG comment / PNG text).
    """

    def __init__(self, config: ImageNetteImageTransformerConfig):
        self.config = config
        self.encoder = create_image_encoder(config.encoding)

    def encode_and_save(self) -> List[SavedEncodedImage]:
        train_loader, val_loader = load_imagenette(
            transform=no_transformer(),
            batch_size=1,
            train_subset_size=-1,
            test_subset_size=-1,
            shuffle=False,
        )
        saved: List[SavedEncodedImage] = []
        saved.extend(self._encode_split("train", train_loader))
        saved.extend(self._encode_split("val", val_loader))
        return saved

    def decode_saved_file(self, path: str) -> torch.Tensor:
        with Image.open(path) as image:
            metadata = self._read_metadata(image)
            params = ImageEncodingParams.from_dict(metadata["encoding"])
            encoded_image = ToTensor()(image.convert("RGB"))
        encoder = create_image_encoder(params)
        decoded = encoder.decode(encoded_image)
        return decoded

    def _encode_split(self, split: str, loader: DataLoader) -> List[SavedEncodedImage]:
        dataset = loader.dataset
        saved: List[SavedEncodedImage] = []
        for index in tqdm(range(len(dataset)), desc=f"Encoding ImageNette {split}"):
            image, label = dataset[index]
            source_path = self._source_path(dataset, index)
            class_name = self._class_name(dataset, int(label))
            encoded = self.encoder.encode(image)
            save_path = self._save_encoded_image(
                split=split,
                encoded_image=encoded,
                label=int(label),
                class_name=class_name,
                source_path=source_path,
                input_shape=list(image.shape),
            )
            saved.append(
                SavedEncodedImage(
                    path=save_path,
                    label=int(label),
                    class_name=class_name,
                    source_path=source_path,
                )
            )
        return saved

    def _save_encoded_image(
        self,
        split: str,
        encoded_image: torch.Tensor,
        label: int,
        class_name: str,
        source_path: str,
        input_shape: List[int],
    ) -> str:
        save_dir = os.path.join(
            self.config.output_root,
            self.config.encoding.strategy.value,
            split,
            class_name,
        )
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, os.path.basename(source_path))
        metadata: Dict = {
            "encoding": self.config.encoding.to_dict(),
            "input_shape": input_shape,
            "label": label,
            "class_name": class_name,
            "source_path": source_path,
            "split": split,
        }
        pil_image = ToPILImage()(encoded_image.detach().cpu().clamp(0.0, 1.0))
        self._save_image_with_metadata(pil_image, save_path, metadata)
        return save_path

    @staticmethod
    def _save_image_with_metadata(
        image: Image.Image,
        save_path: str,
        metadata: Dict,
    ) -> None:
        metadata_json = json.dumps(metadata, ensure_ascii=False)
        ext = os.path.splitext(save_path)[1].lower()
        if ext == ".png":
            png_info = PngImagePlugin.PngInfo()
            png_info.add_text("image_encoding_metadata", metadata_json)
            image.save(save_path, pnginfo=png_info)
            return
        if ext in {".jpg", ".jpeg"}:
            image.save(save_path, comment=metadata_json.encode("utf-8"))
            return
        image.save(save_path)

    @staticmethod
    def _read_metadata(image: Image.Image) -> Dict:
        metadata = image.info.get("image_encoding_metadata")
        if metadata is None:
            metadata = image.info.get("comment")
        if isinstance(metadata, bytes):
            metadata = metadata.decode("utf-8")
        if not metadata:
            raise ValueError("No image encoding metadata found in image file")
        return json.loads(metadata)

    @staticmethod
    def _source_path(dataset, index: int) -> str:
        if isinstance(dataset, Subset):
            original_index = dataset.indices[index]
            return dataset.dataset.samples[original_index][0]
        return dataset.samples[index][0]

    @staticmethod
    def _class_name(dataset, label: int) -> str:
        if isinstance(dataset, Subset):
            return dataset.dataset.classes[label]
        return dataset.classes[label]

def imagenette_image_transformer(
    strategy: ImageEncodingStrategy | str = ImageEncodingStrategy.PERMUTATION,
    output_root: str = "./data/imagenette_encoded",
    seed: int = 42,
    kernel_size: int = 5,
    sigma: float = 1.0,
    bilateral_color_sigma: float = 0.1,
    bilateral_spatial_sigma: float = 1.0,
    iterations: int = 1,
) -> List[SavedEncodedImage]:
    strategy = (
        strategy
        if isinstance(strategy, ImageEncodingStrategy)
        else ImageEncodingStrategy(str(strategy))
    )
    config = ImageNetteImageTransformerConfig(
        output_root=output_root,
        encoding=ImageEncodingParams(
            strategy=strategy,
            seed=seed,
            kernel_size=kernel_size,
            sigma=sigma,
            bilateral_color_sigma=bilateral_color_sigma,
            bilateral_spatial_sigma=bilateral_spatial_sigma,
            iterations=iterations,
        ),
    )
    return ImageNetteImageTransformer(config).encode_and_save()


if __name__ == "__main__":
    strategy = ImageEncodingStrategy.GAUSSIAN_BLUR
    imagenette_image_transformer(strategy=strategy)


    # # Quick manual decode smoke test for the current permutation outputs.
    # # Regenerate permutation images first if this file does not exist yet.
    # encoded_path = os.path.join(
    #     "./data/imagenette_encoded",
    #     "permutation__bilateral_color_sigma=0.1__bilateral_spatial_sigma=1__iterations=1__kernel_size=5__seed=42__sigma=1",
    #     "train",
    #     "n01440764",
    #     "ILSVRC2012_val_00000293.JPEG",
    # )
    # transformer = ImageNetteImageTransformer(ImageNetteImageTransformerConfig())
    # decoded = transformer.decode_saved_file(encoded_path)
    # output_path = "./decoded_permutation_test.JPEG"
    # ToPILImage()(decoded).save(output_path)
    # print(f"Decoded image saved to: {output_path}")

