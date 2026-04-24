import datetime
import os
from typing import Callable, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import ToPILImage

from attacks.attack import Attack
from attacks.attack_factory import AttackFactory
from attacks.attack_names import AttackNames
from config.imagenette_classes import ImageNetteClasses
from data_eng.dataset_loader import load_imagenette
from data_eng.io import load_model_imagenette
from domain.model.model_names import ModelNames
from imagenette_lab.imagenette_direct_attacks import normalize_adversarial_image, save_failure_log


def attack_and_save_images(
    attack: Attack,
    data_loader: DataLoader,
    images_per_class: int = 1,
    successfully_attacked_images_folder: str = "",
    attack_test_dataset: bool = False,
    folder_model_name: Optional[str] = None,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = attack.model
    dataset_name = "test" if attack_test_dataset else "train"

    # create folder if doesn't exist
    if not os.path.exists(successfully_attacked_images_folder):
        os.makedirs(successfully_attacked_images_folder, exist_ok=True)

    model.eval()
    model.to(device)

    # Track attacked images per class
    attacked_images_per_class = {}
    num_classes = len(ImageNetteClasses.get_classes())

    for images, labels in tqdm(data_loader):
        # Check if all classes have exactly the required number of images
        all_classes_complete = all(attacked_images_per_class.get(i, 0) >= images_per_class for i in range(num_classes))
        if all_classes_complete:
            break

        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)

        # Remove missclassified images
        images, labels = images[predictions == labels], labels[predictions == labels]

        if labels.numel() == 0:
            continue

        adv_images = attack(images, labels)

        # Normalize adversarial images to [0,1] range (some attacks may exceed boundaries)
        adv_images = normalize_adversarial_image(adv_images)

        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted_labels = torch.max(outputs.data, 1)
            for i in range(len(adv_images)):
                label = labels[i].item()
                predicted_label = predicted_labels[i].item()

                if predicted_label == label:
                    continue

                # Check if we still need images for this class
                if attacked_images_per_class.get(label, 0) >= images_per_class:
                    continue

                # Increment counter for this class
                attacked_images_per_class[label] = attacked_images_per_class.get(label, 0) + 1

                # Path: successfully_attacked_images_folder/dataset_name/{folder_model_name}/attack_name/label/timestamp.png
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # Include microseconds

                # Create directory structure (folder_model_name aligns with transferability / checkpoint labels)
                name_for_path = folder_model_name if folder_model_name is not None else attack.model_name
                save_dir = os.path.join(successfully_attacked_images_folder, dataset_name, name_for_path, attack.attack, str(label))
                os.makedirs(save_dir, exist_ok=True)

                attcked_image_save_path = os.path.join(save_dir, f"{timestamp}.png")

                # Convert tensors to PIL Images and save as actual PNG files
                to_pil = ToPILImage()

                # Convert adversarial image tensor to PIL Image and save
                adv_pil = to_pil(adv_images[i].cpu())
                adv_pil.save(attcked_image_save_path)

                # Check if all classes have exactly the required number of images
                all_classes_complete = all(attacked_images_per_class.get(i, 0) >= images_per_class for i in range(num_classes))
                if all_classes_complete:
                    print("🎉 All classes completed! Stopping collection.")
                    break


def attack_and_save_images_multiple(
    model_names: List[str],
    attack_names: List[str],
    images_per_class: int = 1,
    successfully_attacked_images_folder: str = "",
    attack_test_dataset: bool = False,
    checkpoint_path_for_model: Optional[Callable[[str], str]] = None,
):
    """
    Args:
        checkpoint_path_for_model: If set, maps architecture name (e.g. resnet18) to a ``.pt`` path.
            Default: ``./models/imagenette/{name}_advanced.pt``
    """

    def _default_checkpoint_path(name: str) -> str:
        return f"./models/imagenette/{name}_advanced.pt"

    resolve = checkpoint_path_for_model or _default_checkpoint_path

    for model_name in model_names:
        model_path = resolve(model_name)
        result = load_model_imagenette(model_path, model_name, device='cuda')
        model = result.model
        folder_label = os.path.splitext(os.path.basename(model_path))[0]
        for attack_name in attack_names:
            print(f"🔍 Running {attack_name} attack on {model_name}...")
            if attack_test_dataset:
                _, data_loader = load_imagenette(batch_size=1, test_subset_size=-1)
            else:
                data_loader, _ = load_imagenette(batch_size=1, train_subset_size=-1)

            attack = AttackFactory.get_attack(attack_name, model)
            try:
                attack_and_save_images(
                    attack,
                    data_loader,
                    images_per_class=images_per_class,
                    successfully_attacked_images_folder=successfully_attacked_images_folder,
                    attack_test_dataset=attack_test_dataset,
                    folder_model_name=folder_label,
                )
            except Exception as e:
                print(f"❌ {attack_name} attack on {model_name} failed: {str(e)}")
                save_failure_log(model_name, attack_name, e, f"{successfully_attacked_images_folder}/{model_name}/{attack_name}")
                continue
            print(f"✅ {attack_name} attack on {model_name} completed!")


if __name__ == "__main__":
    print("\n🎉 Simple example completed!")
    print("📁 Results saved to: results/attacks/imagenette_models/")
    print(len(ImageNetteClasses.get_classes()))

    model_names = [
        ModelNames().resnet18,
        ModelNames().densenet121,
        ModelNames().mobilenet_v2,
        ModelNames().efficientnet_b0,
    ]

    attack_names = AttackNames().all_attack_names

    attack_and_save_images_multiple(
        model_names,
        attack_names,
        images_per_class=10,
        successfully_attacked_images_folder="data/attacks/imagenette_models",
        attack_test_dataset=False,
    )

    attack_and_save_images_multiple(
        model_names,
        attack_names,
        images_per_class=10,
        successfully_attacked_images_folder="data/attacks/imagenette_models",
        attack_test_dataset=True,
    )
