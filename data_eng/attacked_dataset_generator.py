from typing import List
from torch.nn import Module
from tqdm import tqdm

from attacks.simple_attacks import SimpleAttacks
from data_eng.dataset_loader import DatasetLoader, DatasetType
from torch.utils.data import DataLoader, ConcatDataset


class AttackedDatasetGenerator:
    """
    Class for generating attacked datasets from original imagenette dataset.
    """

    @staticmethod
    def get_attacked_imagenette_dataset_from_attack(
      model: Module,
      model_name: str,
      attack_name: str,
      num_of_images: int,
      attack_ratio: float,
      use_test_set: bool = False,
      shuffle=True):
        """
        Returns attacked imagenette dataset.
        :param model: model to attack
        :param model_name: model name
        :param attack_name: attack name
        :param batch_size: batch size
        :param shuffle: shuffle
        :return: attacked imagenette dataset
        """

        attacked_images_count = int(num_of_images * attack_ratio)
        attacked_results = SimpleAttacks.get_attacked_imagenette_images(
          attack_name=attack_name,
          model=model,
          model_name=model_name,
          num_of_images=attacked_images_count,
          use_test_set=use_test_set,
          batch_size=1)

        attacked_images = [(res.adv_image, res.label) for res in attacked_results]
        train_loader, test_loader = DatasetLoader.get_dataset_by_type(dataset_type=DatasetType.IMAGENETTE, batch_size=1, shuffle=shuffle)
        data_loader = test_loader if use_test_set else train_loader

        clean_images_count = num_of_images - attacked_images_count
        clean_images = []
        for i, (images, labels) in enumerate(data_loader):
            if i > clean_images_count:
                break

            for j in range(len(images)):
              clean_images.append((images[j], labels[j]))

        return clean_images + attacked_images

    @staticmethod
    def get_attacked_imagenette_dataset(
      model: Module,
      model_name: str,
      attack_names: List['str'],
      num_of_images_per_attack: int,
      attack_ratio: float,
      use_test_set: bool = False,
      batch_size: int = 1,
      shuffle = True
    ) -> DataLoader:

      datasets = []

      loading_bar = tqdm(attack_names, desc='Generating attacked datasets')
      for attack_name in attack_names:
        loading_bar.set_description(f'Generating attacked datasets - {attack_name}')
        dataset = AttackedDatasetGenerator.get_attacked_imagenette_dataset_from_attack(
          model=model,
          model_name=model_name,
          attack_name=attack_name,
          num_of_images=num_of_images_per_attack,
          attack_ratio=attack_ratio,
          use_test_set=use_test_set,
          shuffle=shuffle
        )
        datasets.extend(dataset)

        loading_bar.update()

      return DataLoader(datasets, batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    def get_attacked_imagenette_dataset_half(model: Module, attack_names: List['str'], model_name=None, test_subset_size: int = 1000, batch_size: int = 1) -> DataLoader:
      _, test_imagenette_loader = DatasetLoader.get_dataset_by_type(dataset_type=DatasetType.IMAGENETTE, batch_size=batch_size, test_subset_size=test_subset_size)
      images_per_attack = int(len(test_imagenette_loader.dataset) / len(attack_names))
      attacked_results: DataLoader = AttackedDatasetGenerator.get_attacked_imagenette_dataset(
          model=model,
          attack_names=attack_names,
          attack_ratio=1.0,
          num_of_images_per_attack=images_per_attack,
          batch_size=1,
          use_test_set=True,
          model_name=model_name
      )

      merged_dataset = ConcatDataset([test_imagenette_loader.dataset, attacked_results.dataset])
      return DataLoader(merged_dataset, batch_size=batch_size, shuffle=True)