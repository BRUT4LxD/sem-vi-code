import random

from torch.utils.data import ConcatDataset, DataLoader, Dataset

class BinaryLabeledSubsetDataset(Dataset):
    def __init__(self, base_dataset, indices, label: int) -> None:
        self.base_dataset = base_dataset
        self.indices = indices
        self.label = label

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, _ = self.base_dataset[self.indices[idx]]
        return image, self.label


def build_binary_noise_loader(
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

    clean_binary_dataset = BinaryLabeledSubsetDataset(clean_dataset, clean_indices, 0)
    attacked_binary_dataset = BinaryLabeledSubsetDataset(attacked_dataset, attacked_indices, 1)
    mixed_dataset = ConcatDataset([attacked_binary_dataset, clean_binary_dataset])

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
