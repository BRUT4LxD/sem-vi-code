from ast import Tuple
from typing import List
from torch.nn import Module
from tqdm import tqdm
from attacks.attack_factory import AttackFactory

from attacks.simple_attacks import SimpleAttacks
from config.imagenet_models import ImageNetModels
from data_eng.dataset_loader import DatasetLoader, DatasetType, load_imagenette
from torch.utils.data import DataLoader, ConcatDataset

from domain.attack_distance_score import AttackDistanceScore
from domain.attack_result import AttackedImageResult

class AttackedDatasetGeneratorResult:
    def __init__(self, data_loader: DataLoader, attack_distance_score: AttackDistanceScore) -> None:
        self.data_loader = data_loader
        self.attack_distance_score = attack_distance_score

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
            if i >= clean_images_count:
                break

            for j in range(len(images)):
              clean_images.append((images[j], labels[j]))

        l1, l2, lInf, power = 0, 0, 0, 0
        for res in attacked_results:
          l1 += res.distance.l1
          l2 += res.distance.l2
          lInf += res.distance.linf
          power += res.distance.power

        average_distance_score = AttackDistanceScore(l1=l1 / len(attacked_results), l2=l2 / len(attacked_results), linf=lInf / len(attacked_results), power=power / len(attacked_results))

        return clean_images + attacked_images, average_distance_score

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
    ) -> AttackedDatasetGeneratorResult:

      datasets = []

      loading_bar = tqdm(attack_names, desc='Generating attacked datasets')
      for attack_name in attack_names:
        loading_bar.set_description(f'Generating attacked datasets - {attack_name}')
        dataset, distance_score = AttackedDatasetGenerator.get_attacked_imagenette_dataset_from_attack(
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

      return AttackedDatasetGeneratorResult(DataLoader(datasets, batch_size=batch_size, shuffle=shuffle), distance_score)

    @staticmethod
    def get_attacked_imagenette_dataset_half(model: Module, attack_names: List['str'], model_name=None, test_subset_size: int = -1, batch_size: int = 1) -> DataLoader:
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


    @staticmethod
    def get_attacked_imagenette_dataset_multimodel(
      model_names: List[str],
      attack_names: List['str'],
      num_of_images_per_attack: int,
      use_test_set: bool = True,
      batch_size: int = 1,
      device: str = 'cuda') -> DataLoader:
      """
      Returns attacked imagenette dataset for all model names.
      :param model_names: list of model names
      :param attack_names: list of attack names
      :param num_of_images_per_attack: number of images per attack
      :param use_test_set: use test set
      :param batch_size: batch size of returned dataloader
      :param device: device
      :return: Dataloader
      """
      attacked_dataset: list['AttackedImageResult'] = []
      for model_name in tqdm(model_names):
        current_model = ImageNetModels().get_model(model_name=model_name)
        current_model = current_model.to(device)

        for attack_name in tqdm(attack_names):
          attacked_subset = SimpleAttacks.get_attacked_imagenette_images(
            model=current_model,
            attack_name=attack_name,
            model_name=model_name,
            batch_size=1,
            num_of_images=num_of_images_per_attack,
            use_test_set=use_test_set
          )

          attacked_dataset.extend(attacked_subset)


      att_dataset = [(att.adv_image, att.label) for att in attacked_dataset]
      return DataLoader(att_dataset, batch_size=batch_size, shuffle=True)


    @staticmethod
    def get_attacked_imagenette_dataset_multimodel_for_binary(
      model_names: List[str],
      attack_names: List['str'],
      num_of_images_per_attack: int,
      use_test_set: bool = True,
      batch_size: int = 1,
      device: str = 'cuda') -> DataLoader:
      """
      Returns attacked imagenette dataset for all model names.
      :param model_names: list of model names
      :param attack_names: list of attack names
      :param num_of_images_per_attack: number of images per attack
      :param use_test_set: use test set
      :param batch_size: batch size of returned dataloader
      :param device: device
      :return: Dataloader
      """

      inner_batch_size = 1

      attacked_dataset: list['AttackedImageResult'] = []
      for model_name in tqdm(model_names):
        current_model = ImageNetModels().get_model(model_name=model_name)
        current_model = current_model.to(device)

        for attack_name in tqdm(attack_names):
          attacked_subset = SimpleAttacks.get_attacked_imagenette_images(
            model=current_model,
            attack_name=attack_name,
            model_name=model_name,
            batch_size=inner_batch_size,
            num_of_images=num_of_images_per_attack,
            use_test_set=use_test_set
          )

          attacked_dataset.extend(attacked_subset)


      att_dataset = [(att.adv_image, 1) for att in attacked_dataset]
      source_dataset = []

      train_loader, test_loader = load_imagenette(batch_size=inner_batch_size, shuffle=True)
      dataloader = test_loader if use_test_set else train_loader

      i = 0
      for images, labels in dataloader:
        if i == len(att_dataset):
          break

        source_dataset.append((images[0], 0))
        i += 1

      return DataLoader(att_dataset + source_dataset, batch_size=batch_size, shuffle=True)
