import random
from typing import Optional

import torchvision.datasets as datasets
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from data_eng.dataset_loader import load_attacked_imagenette
from data_eng.transforms import imagenette_transformer

ATTACKED_IMAGENETTE_FOLDER = "data/attacks/imagenette_models"


def _default_clean_train_root() -> str:
    return "./data/imagenette/train"


def _default_clean_val_root() -> str:
    return "./data/imagenette/val"

class IndexedSubsetDataset(Dataset):
    def __init__(self, base_dataset, indices) -> None:
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]


def build_adversarial_training_loader(
    clean_dataset,
    attacked_dataset,
    batch_size: int,
    shuffle: bool,
    split_name: str,
    clean_to_attacked_ratio: float = 1.0,
    augment_clean_to_match_attacked: bool = True,
    random_seed: int = 42,
) -> DataLoader:
    if len(clean_dataset) == 0 or len(attacked_dataset) == 0:
        raise FileNotFoundError(
            f"Unable to build {split_name} split: missing clean or attacked images"
        )

    if clean_to_attacked_ratio == -1:
        clean_count = len(clean_dataset)
        attacked_count = len(attacked_dataset)
    else:
        if clean_to_attacked_ratio <= 0:
            raise ValueError(
                "clean_to_attacked_ratio must be positive or -1 to disable balancing"
            )

        if augment_clean_to_match_attacked:
            attacked_count = len(attacked_dataset)
            clean_count = max(1, int(attacked_count * clean_to_attacked_ratio))
        else:
            attacked_count = min(
                len(attacked_dataset),
                max(1, int(len(clean_dataset) / clean_to_attacked_ratio)),
            )
            clean_count = min(
                len(clean_dataset),
                max(1, int(attacked_count * clean_to_attacked_ratio)),
            )

    rng = random.Random(random_seed)
    attacked_indices = rng.sample(range(len(attacked_dataset)), attacked_count)

    if augment_clean_to_match_attacked and clean_count > len(clean_dataset):
        clean_indices = rng.choices(range(len(clean_dataset)), k=clean_count)
    else:
        clean_indices = rng.sample(range(len(clean_dataset)), clean_count)

    clean_subset = IndexedSubsetDataset(clean_dataset, clean_indices)
    attacked_subset = IndexedSubsetDataset(attacked_dataset, attacked_indices)
    mixed_dataset = ConcatDataset([attacked_subset, clean_subset])

    print(
        f"📊 {split_name.capitalize()} split: {len(mixed_dataset)} total "
        f"({attacked_count} attacked + {clean_count} clean)"
    )
    if augment_clean_to_match_attacked and clean_count > len(clean_dataset):
        print(
            f"   Clean augmentation enabled: expanded {len(clean_dataset)} clean images "
            f"to {clean_count} samples"
        )

    return DataLoader(mixed_dataset, batch_size=batch_size, shuffle=shuffle)


def build_imagenette_adversarial_training_loaders(
    batch_size: int,
    attacked_subset_size: int = -1,
    augment_clean_to_match_attacked: bool = True,
    clean_to_attacked_ratio: float = 1.0,
    train_test_split: Optional[float] = None,
    random_seed: int = 42,
    attacked_images_folder: str = ATTACKED_IMAGENETTE_FOLDER,
    clean_train_root: Optional[str] = None,
    clean_val_root: Optional[str] = None,
):
    attacked_train_loader, attacked_test_loader = load_attacked_imagenette(
        path_to_data=attacked_images_folder,
        batch_size=batch_size,
        train_subset_size=attacked_subset_size,
        test_subset_size=attacked_subset_size,
        shuffle=False,
    )

    transform = _resolve_dataset_transform(attacked_train_loader.dataset)
    train_root = clean_train_root or _default_clean_train_root()
    val_root = clean_val_root or _default_clean_val_root()
    clean_train_dataset = datasets.ImageFolder(
        root=train_root,
        transform=transform,
    )
    clean_test_dataset = datasets.ImageFolder(
        root=val_root,
        transform=transform,
    )

    attacked_train_dataset = attacked_train_loader.dataset
    attacked_test_dataset = attacked_test_loader.dataset

    if train_test_split is not None:
        if not (0.0 < train_test_split <= 1.0):
            raise ValueError("train_test_split must be between 0 (exclusive) and 1 (inclusive)")
        max_test_attacked = max(1, int(len(attacked_train_dataset) * train_test_split))
        max_test_clean = max(1, int(len(clean_train_dataset) * train_test_split))
        attacked_test_dataset = _cap_dataset(attacked_test_dataset, max_test_attacked, random_seed)
        clean_test_dataset = _cap_dataset(clean_test_dataset, max_test_clean, random_seed)

    train_loader = build_adversarial_training_loader(
        clean_dataset=clean_train_dataset,
        attacked_dataset=attacked_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        split_name="train",
        clean_to_attacked_ratio=clean_to_attacked_ratio,
        augment_clean_to_match_attacked=augment_clean_to_match_attacked,
        random_seed=random_seed,
    )
    test_loader = build_adversarial_training_loader(
        clean_dataset=clean_test_dataset,
        attacked_dataset=attacked_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        split_name="test",
        clean_to_attacked_ratio=clean_to_attacked_ratio,
        augment_clean_to_match_attacked=augment_clean_to_match_attacked,
        random_seed=random_seed,
    )

    return train_loader, test_loader


def _cap_dataset(dataset, max_size: int, random_seed: int):
    if len(dataset) <= max_size:
        return dataset
    rng = random.Random(random_seed)
    indices = rng.sample(range(len(dataset)), max_size)
    return Subset(dataset, indices)


def _resolve_dataset_transform(dataset):
    if hasattr(dataset, "datasets") and len(getattr(dataset, "datasets", [])) > 0:
        return _resolve_dataset_transform(dataset.datasets[0])
    current = dataset
    while current is not None:
        transform = getattr(current, "transform", None)
        if transform is not None:
            return transform
        current = getattr(current, "dataset", None)
    return imagenette_transformer()
