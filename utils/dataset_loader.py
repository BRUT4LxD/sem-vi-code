import torchvision.transforms as transforms
import torchvision.datasets as datasets


def load_MNIST(transform, path_to_data='./data'):
    trans = transform if transform is not None else transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=path_to_data, train=True, download=True, transform=trans)
    test_dataset = datasets.MNIST(
        root=path_to_data, train=False, download=True, transform=trans)

    return train_dataset, test_dataset


def load_CIFAR10(transform, path_to_data='./data'):
    trans = transform if transform is not None else transforms.ToTensor()

    train_dataset = datasets.CIFAR10(
        root=path_to_data, train=True, download=True, transform=trans)
    test_dataset = datasets.CIFAR10(
        root=path_to_data, train=False, download=True, transform=trans)

    return train_dataset, test_dataset


def load_imagenette(transform, path_to_data='./data/imagenette'):
    trans = transform if transform is not None else transforms.ToTensor()

    train_path = path_to_data + '/train'
    test_path = path_to_data + '/val'

    train_dataset = datasets.ImageFolder(root=train_path, transform=trans)
    test_dataset = datasets.ImageFolder(root=test_path, transform=trans)

    return train_dataset, test_dataset
