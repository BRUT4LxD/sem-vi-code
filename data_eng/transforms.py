import torchvision.transforms as transforms


def mnist_transformer():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ]
    )


def cifar_transformer():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )


def imagenette_transformer():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
    )
