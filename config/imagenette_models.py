
from typing import List
from architectures.resnet import ResNet18, ResNet50, ResNet101, ResNet152
from architectures.densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201
from architectures.efficientnet import EfficientNetB0
from architectures.mobilenetv2 import MobileNetV2
from architectures.vgg import VGG11, VGG13, VGG16, VGG19

from domain.model_config import ModelConfig


def get_imagenette_pretrained_models() -> List['ModelConfig']:
    return [
        ModelConfig(ResNet18(), 'models/resnet18imagenette.pt', True),
        ModelConfig(ResNet50(), 'models/resnet50imagenette.pt', True),
        ModelConfig(ResNet101(), 'models/resnet101imagenette.pt', True),
        ModelConfig(ResNet152(), 'models/resnet152imagenette.pt', True),
        ModelConfig(DenseNet121(), 'models/densenet121imagenette.pt', True),
        ModelConfig(DenseNet161(), 'models/densenet161imagenette.pt', True),
        ModelConfig(DenseNet169(), 'models/densenet169imagenette.pt', True),
        ModelConfig(DenseNet201(), 'models/densenet201imagenette.pt', True),
        ModelConfig(EfficientNetB0(),
                    'models/efficientnetb0imagenette.pt', True),
        ModelConfig(MobileNetV2(), 'models/mobilenetv2imagenette.pt', True),
        ModelConfig(VGG11(), 'models/vgg11imagenette.pt', True),
        ModelConfig(VGG13(), 'models/vgg13imagenette.pt', True),
        ModelConfig(VGG16(), 'models/vgg16imagenette.pt', True),
        ModelConfig(VGG19(), 'models/vgg19imagenette.pt', True),
    ]
