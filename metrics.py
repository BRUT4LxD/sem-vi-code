import torch


def l1_distance(image1, image2):
    return torch.norm(image1 - image2, p=1)


def l2_distance(image1, image2):
    return torch.norm(image1 - image2, p=2)


def linf_distance(image1, image2):
    return torch.norm(image1 - image2, p=float('inf'))
