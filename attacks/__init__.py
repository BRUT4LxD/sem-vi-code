from attacks.black_box import get_all_black_box_attack
from attacks.white_box import get_all_white_box_attack
import torch


def get_all_attacks(model: torch.nn.Module):
    black_box = get_all_black_box_attack(model)
    white_box = get_all_white_box_attack(model)

    all_attacks = []
    all_attacks.extend(black_box)
    all_attacks.extend(white_box)

    return all_attacks
