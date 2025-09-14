import torch

from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, resnet101, ResNet101_Weights, resnet152, ResNet152_Weights
from torchvision.models import densenet121, DenseNet121_Weights, densenet161, DenseNet161_Weights, densenet169, DenseNet169_Weights, densenet201, DenseNet201_Weights
from torchvision.models import vgg11, VGG11_Weights, vgg13, VGG13_Weights, vgg16, VGG16_Weights, vgg19, VGG19_Weights
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights, mobilenet_v3_small, MobileNet_V3_Small_Weights, mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights, efficientnet_b2, EfficientNet_B2_Weights, efficientnet_b3, EfficientNet_B3_Weights, efficientnet_b4, EfficientNet_B4_Weights, efficientnet_b5, EfficientNet_B5_Weights, efficientnet_b6, EfficientNet_B6_Weights, efficientnet_b7, EfficientNet_B7_Weights
from domain.model.model_names import ModelNames

class PretrainedModelDownloader:

  @staticmethod
  def download_resnet_models(return_model: bool = False) -> list:
    r1 = resnet18(weights=ResNet18_Weights.DEFAULT)
    r2 = resnet50(weights=ResNet50_Weights.DEFAULT)
    r3 = resnet101(weights=ResNet101_Weights.DEFAULT)
    r4 = resnet152(weights=ResNet152_Weights.DEFAULT)

    if return_model:
      return [r1, r2, r3, r4]

    return []

  @staticmethod
  def download_densenet_models(return_model: bool = False) -> list:
    d1 = densenet121(weights=DenseNet121_Weights.DEFAULT)
    d2 = densenet161(weights=DenseNet161_Weights.DEFAULT)
    d3 = densenet169(weights=DenseNet169_Weights.DEFAULT)
    d4 = densenet201(weights=DenseNet201_Weights.DEFAULT)

    if return_model:
      return [d1, d2, d3, d4]

    return []

  @staticmethod
  def download_vgg_models(return_model: bool = False):
    v1 = vgg11(weights=VGG11_Weights.DEFAULT)
    v2 = vgg13(weights=VGG13_Weights.DEFAULT)
    v3 = vgg16(weights=VGG16_Weights.DEFAULT)
    v4 = vgg19(weights=VGG19_Weights.DEFAULT)

    if return_model:
      return [v1, v2, v3, v4]

    return []

  @staticmethod
  def download_mobilenet_models(return_model: bool = False):
    m1 = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    m2 = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    m3 = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

    if return_model:
      return [m1, m2, m3]

    return []

  @staticmethod
  def download_efficientnet_models(return_model: bool = False):
    e0 = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    e1 = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
    e2 = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
    e3 = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
    e4 = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
    e5 = efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
    e6 = efficientnet_b6(weights=EfficientNet_B6_Weights.DEFAULT)
    e7 = efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)

    if return_model:
      return [e0, e1, e2, e3, e4, e5, e6, e7]

    return []

  @staticmethod
  def download_all_models(return_model: bool = False):
    r = PretrainedModelDownloader.download_resnet_models(return_model=return_model)
    d = PretrainedModelDownloader.download_densenet_models(return_model=return_model)
    v = PretrainedModelDownloader.download_vgg_models(return_model=return_model)
    m = PretrainedModelDownloader.download_mobilenet_models(return_model=return_model)
    e = PretrainedModelDownloader.download_efficientnet_models(return_model=return_model)

    if return_model:
      return r + d + v + m + e

    return []

  @staticmethod
  def download_model(name: str) -> torch.nn.Module:
    if name == ModelNames.resnet18:
      return resnet18(weights=ResNet18_Weights.DEFAULT)

    if name == ModelNames.resnet50:
      return resnet50(weights=ResNet50_Weights.DEFAULT)

    if name == ModelNames.resnet101:
      return resnet101(weights=ResNet101_Weights.DEFAULT)

    if name == ModelNames.resnet152:
      return resnet152(weights=ResNet152_Weights.DEFAULT)

    if name == ModelNames.densenet121:
      return densenet121(weights=DenseNet121_Weights.DEFAULT)

    if name == ModelNames.densenet161:
      return densenet161(weights=DenseNet161_Weights.DEFAULT)

    if name == ModelNames.densenet169:
      return densenet169(weights=DenseNet169_Weights.DEFAULT)

    if name == ModelNames.densenet201:
      return densenet201(weights=DenseNet201_Weights.DEFAULT)

    if name == ModelNames.vgg11:
      return vgg11(weights=VGG11_Weights.DEFAULT)

    if name == ModelNames.vgg13:
      return vgg13(weights=VGG13_Weights.DEFAULT)

    if name == ModelNames.vgg16:
      return vgg16(weights=VGG16_Weights.DEFAULT)

    if name == ModelNames.vgg19:
      return vgg19(weights=VGG19_Weights.DEFAULT)

    if name == ModelNames.mobilenet_v2:
      return mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    if name == ModelNames.mobilenet_v3_small:
      return mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

    if name == ModelNames.mobilenet_v3_large:
      return mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

    if name == ModelNames.efficientnet_b0:
      return efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    if name == ModelNames.efficientnet_b1:
      return efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)

    if name == ModelNames.efficientnet_b2:
      return efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)

    if name == ModelNames.efficientnet_b3:
      return efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)

    if name == ModelNames.efficientnet_b4:
      return efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)

    if name == ModelNames.efficientnet_b5:
      return efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)

    if name == ModelNames.efficientnet_b6:
      return efficientnet_b6(weights=EfficientNet_B6_Weights.DEFAULT)

    if name == ModelNames.efficientnet_b7:
      return efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)

    raise ValueError(f"Model name not found: {name}")


if __name__ == "__main__":
  PretrainedModelDownloader.download_all_models()

