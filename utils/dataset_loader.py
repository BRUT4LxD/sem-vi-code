import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from torch.utils.data import DataLoader, Subset

from utils.transforms import cifar_transformer, imagenette_transformer, mnist_transformer


def _get_data_loaders(train_dataset, test_dataset, batch_size=1, train_subset_size=-1, test_subset_size=-1):
    train_loader = None
    test_loader = None

    if train_subset_size > 0:
        subset = Subset(train_dataset, list(range(train_subset_size)))
        train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)

    if test_subset_size > 0:
        subset = Subset(test_dataset, list(range(test_subset_size)))
        test_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    else:
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True)

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


def load_imagenette(transform=None, path_to_data='./data/imagenette', batch_size=1, train_subset_size=-1, test_subset_size=-1):
    trans = transform if transform is not None else imagenette_transformer()

    train_path = path_to_data + '/train'
    test_path = path_to_data + '/val'

    train_dataset = datasets.ImageFolder(root=train_path, transform=trans)
    test_dataset = datasets.ImageFolder(root=test_path, transform=trans)

    return _get_data_loaders(train_dataset, test_dataset, batch_size, train_subset_size, test_subset_size)
