import torchvision
import torchvision.transforms as transforms

def load_MNIST(transform, path_to_data='./data'):
    trans = transform if transform is not None else transforms.ToTensor()

    train_dataset = torchvision.datasets.MNIST(
        root=path_to_data, train=True, download=True, transform=trans)
    test_dataset = torchvision.datasets.MNIST(
        root=path_to_data, train=False, download=True, transform=trans)

    return train_dataset, test_dataset


def load_CIFAR10(transform, path_to_data='./data'):
    trans = transform if transform is not None else transforms.ToTensor()

    train_dataset = torchvision.datasets.CIFAR10(
        root=path_to_data, train=True, download=True, transform=trans)
    test_dataset = torchvision.datasets.CIFAR10(
        root=path_to_data, train=False, download=True, transform=trans)

    return train_dataset, test_dataset
