import torch

from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, resnet101, ResNet101_Weights, resnet152, ResNet152_Weights
from torchvision.models import densenet121, DenseNet121_Weights, densenet161, DenseNet161_Weights, densenet169, DenseNet169_Weights, densenet201, DenseNet201_Weights
from torchvision.models import vgg11, VGG11_Weights, vgg13, VGG13_Weights, vgg16, VGG16_Weights, vgg19, VGG19_Weights
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights, mobilenet_v3_small, MobileNet_V3_Small_Weights, mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights, efficientnet_b2, EfficientNet_B2_Weights, efficientnet_b3, EfficientNet_B3_Weights, efficientnet_b4, EfficientNet_B4_Weights, efficientnet_b5, EfficientNet_B5_Weights, efficientnet_b6, EfficientNet_B6_Weights, efficientnet_b7, EfficientNet_B7_Weights
from domain.model_names import ModelNames

class PretrainedModelDownloader:

  @staticmethod
  def download_resnet_models():
    resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet50(weights=ResNet50_Weights.DEFAULT)
    resnet101(weights=ResNet101_Weights.DEFAULT)
    resnet152(weights=ResNet152_Weights.DEFAULT)

  @staticmethod
  def download_densenet_models():
    densenet121(weights=DenseNet121_Weights.DEFAULT)
    densenet161(weights=DenseNet161_Weights.DEFAULT)
    densenet169(weights=DenseNet169_Weights.DEFAULT)
    densenet201(weights=DenseNet201_Weights.DEFAULT)

  @staticmethod
  def download_vgg_models(): 
    vgg11(weights=VGG11_Weights.DEFAULT)
    vgg13(weights=VGG13_Weights.DEFAULT)
    vgg16(weights=VGG16_Weights.DEFAULT)
    vgg19(weights=VGG19_Weights.DEFAULT)

  @staticmethod
  def download_mobilenet_models():
    mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

  @staticmethod
  def download_efficientnet_models():
    efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
    efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
    efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
    efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
    efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
    efficientnet_b6(weights=EfficientNet_B6_Weights.DEFAULT)
    efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)

  @staticmethod
  def download_all_models():
    PretrainedModelDownloader.download_resnet_models()
    PretrainedModelDownloader.download_densenet_models()
    PretrainedModelDownloader.download_vgg_models()
    PretrainedModelDownloader.download_mobilenet_models()
    PretrainedModelDownloader.download_efficientnet_models()

  @staticmethod
  def download_model(name: str) -> torch.nn.Module:
    if not ModelNames.is_valid_model_name(name):
      raise ValueError(f"Model name not found: {name}")

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

