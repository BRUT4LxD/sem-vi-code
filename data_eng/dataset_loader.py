import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler, Dataset
from attacks.attack_names import AttackNames
from data_eng.transforms import mnist_transformer, cifar_transformer, imagenette_transformer, no_transformer
from domain.model.model_names import ModelNames
import torch
import random

class DatasetType (object):
    CIFAR10 = 'cifar10'
    IMAGENETTE = 'imagenette'
    MNIST = 'mnist'

    @staticmethod
    def is_valid_data_set_type(data_set_type):
        return data_set_type in [DatasetType.CIFAR10, DatasetType.IMAGENETTE, DatasetType.MNIST]

class EmptyDataset(Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return None

class DatasetLoader:
    @staticmethod
    def get_dataset_by_type(dataset_type: str, transform=None, batch_size=1, train_subset_size=-1, test_subset_size=-1, shuffle=True):
        if not DatasetType.is_valid_data_set_type(dataset_type):
            raise Exception('Invalid dataset type')

        if dataset_type == DatasetType.CIFAR10:
            return load_CIFAR10(transform, batch_size=batch_size, train_subset_size=train_subset_size, test_subset_size=test_subset_size)

        if dataset_type == DatasetType.MNIST:
            return load_MNIST(transform, batch_size=batch_size, train_subset_size=train_subset_size, test_subset_size=test_subset_size)

        if dataset_type == DatasetType.IMAGENETTE:
            return load_imagenette(transform, batch_size=batch_size, train_subset_size=train_subset_size, test_subset_size=test_subset_size, shuffle=shuffle)

    @staticmethod
    def get_attacked_imagenette_dataset(model_name: str, attack_name: str, batch_size=1, shuffle=True, train_base_path=None, test_base_path=None, pt_extension=False):
        if model_name not in ModelNames().all_model_names:
            raise Exception('Invalid model name')

        if attack_name not in AttackNames().all_attack_names:
            raise Exception('Invalid attack name')

        train_base_path = train_base_path if train_base_path is not None else f'./data/attacked_imagenette_train'
        test_base_path = test_base_path if test_base_path is not None else f'./data/attacked_imagenette_test'

        def tensor_transformer(tensor):
            return tensor

        # load attacked imagenette from folder `attacked_imagenette_train/{model_name}/{attack_name}`
        if pt_extension:
            train_data = datasets.DatasetFolder(root=f'{train_base_path}/{model_name}/{attack_name}', loader=torch.load, transform=tensor_transformer, extensions=('.pt',), target_transform=None)
            test_data = datasets.DatasetFolder(root=f'{test_base_path}/{model_name}/{attack_name}', loader=torch.load, transform=tensor_transformer, extensions=('.pt',), target_transform=None)
        else:
            train_data = datasets.ImageFolder(root=f'{train_base_path}/{model_name}/{attack_name}', transform=no_transformer())
            test_data = datasets.ImageFolder(root=f'{test_base_path}/{model_name}/{attack_name}', transform=no_transformer())

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)

        return train_loader, test_loader


def _get_data_loaders(train_dataset, test_dataset, batch_size=1, train_subset_size=-1, test_subset_size=-1, shuffle=True, random_seed=42):
    train_loader = None
    test_loader = None

    if train_subset_size > 0:
        # Use random sampling instead of taking first N samples
        random.seed(random_seed)
        train_indices = random.sample(range(len(train_dataset)), min(train_subset_size, len(train_dataset)))
        subset = Subset(train_dataset, train_indices)
        train_loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle)

    if test_subset_size > 0:
        # Use random sampling instead of taking first N samples
        random.seed(random_seed)  # Use same seed for reproducibility
        test_indices = random.sample(range(len(test_dataset)), min(test_subset_size, len(test_dataset)))
        subset = Subset(test_dataset, test_indices)
        test_loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
    else:
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader


def load_MNIST(transform=None, path_to_data='./data', batch_size=1, train_subset_size=-1, test_subset_size=-1):
    trans = transform if transform is not None else mnist_transformer()

    train_dataset = datasets.MNIST(
        root=path_to_data, train=True, download=True, transform=trans)
    test_dataset = datasets.MNIST(
        root=path_to_data, train=False, download=True, transform=trans)

    return _get_data_loaders(train_dataset, test_dataset, batch_size, train_subset_size, test_subset_size)


def load_CIFAR10(transform=None, path_to_data='./data', batch_size=1, train_subset_size=-1, test_subset_size=-1):
    trans = transform if transform is not None else cifar_transformer()

    train_dataset = datasets.CIFAR10(
        root=path_to_data, train=True, download=True, transform=trans)
    test_dataset = datasets.CIFAR10(
        root=path_to_data, train=False, download=True, transform=trans)

    return _get_data_loaders(train_dataset, test_dataset, batch_size, train_subset_size, test_subset_size)


def load_imagenette(transform=None, path_to_data='./data/imagenette', batch_size=1, train_subset_size=-1, test_subset_size=-1, shuffle=True):
    trans = transform if transform is not None else imagenette_transformer()

    train_path = path_to_data + '/train'
    test_path = path_to_data + '/val'

    train_dataset = datasets.ImageFolder(root=train_path, transform=trans)
    test_dataset = datasets.ImageFolder(root=test_path, transform=trans)

    return _get_data_loaders(
        train_dataset,
        test_dataset,
        batch_size=batch_size,
        train_subset_size=train_subset_size,
        test_subset_size=test_subset_size,
        shuffle=shuffle)

def load_empty_dataloader():
    return DataLoader(EmptyDataset(), batch_size=1)