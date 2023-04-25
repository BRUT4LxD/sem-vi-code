from .pixle import Pixle
from .square import Square


def get_all_black_box_attack(model):
    return [
        Pixle(model),
        Square(model),
    ]
