from __future__ import annotations

import torch
from typing import Callable, Dict, List

from domain.model.model_names import ModelNames
from torchvision.models import (
  DenseNet121_Weights,
  DenseNet161_Weights,
  DenseNet169_Weights,
  DenseNet201_Weights,
  EfficientNet_B0_Weights,
  EfficientNet_B1_Weights,
  EfficientNet_B2_Weights,
  EfficientNet_B3_Weights,
  EfficientNet_B4_Weights,
  EfficientNet_B5_Weights,
  EfficientNet_B6_Weights,
  EfficientNet_B7_Weights,
  EfficientNet_V2_L_Weights,
  EfficientNet_V2_M_Weights,
  EfficientNet_V2_S_Weights,
  Inception_V3_Weights,
  MaxVit_T_Weights,
  MobileNet_V2_Weights,
  MobileNet_V3_Large_Weights,
  MobileNet_V3_Small_Weights,
  ResNet101_Weights,
  ResNet152_Weights,
  ResNet18_Weights,
  ResNet34_Weights,
  ResNet50_Weights,
  Swin_B_Weights,
  Swin_S_Weights,
  Swin_T_Weights,
  Swin_V2_B_Weights,
  Swin_V2_S_Weights,
  Swin_V2_T_Weights,
  ViT_B_16_Weights,
  ViT_B_32_Weights,
  ViT_L_16_Weights,
  ViT_L_32_Weights,
  densenet121,
  densenet161,
  densenet169,
  densenet201,
  efficientnet_b0,
  efficientnet_b1,
  efficientnet_b2,
  efficientnet_b3,
  efficientnet_b4,
  efficientnet_b5,
  efficientnet_b6,
  efficientnet_b7,
  efficientnet_v2_l,
  efficientnet_v2_m,
  efficientnet_v2_s,
  inception_v3,
  maxvit_t,
  mobilenet_v2,
  mobilenet_v3_large,
  mobilenet_v3_small,
  resnet101,
  resnet152,
  resnet18,
  resnet34,
  resnet50,
  swin_b,
  swin_s,
  swin_t,
  swin_v2_b,
  swin_v2_s,
  swin_v2_t,
  vit_b_16,
  vit_b_32,
  vit_l_16,
  vit_l_32,
)


def _download_inception_v3() -> torch.nn.Module:
  model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
  model.aux_logits = False
  model.AuxLogits = None
  return model


_mn = ModelNames()
_PRETRAINED_LOADERS: Dict[str, Callable[[], torch.nn.Module]] = {
  _mn.densenet121: lambda: densenet121(weights=DenseNet121_Weights.DEFAULT),
  _mn.densenet161: lambda: densenet161(weights=DenseNet161_Weights.DEFAULT),
  _mn.densenet169: lambda: densenet169(weights=DenseNet169_Weights.DEFAULT),
  _mn.densenet201: lambda: densenet201(weights=DenseNet201_Weights.DEFAULT),
  _mn.efficientnet_b0: lambda: efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT),
  _mn.efficientnet_b1: lambda: efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT),
  _mn.efficientnet_b2: lambda: efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT),
  _mn.efficientnet_b3: lambda: efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT),
  _mn.efficientnet_b4: lambda: efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT),
  _mn.efficientnet_b5: lambda: efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT),
  _mn.efficientnet_b6: lambda: efficientnet_b6(weights=EfficientNet_B6_Weights.DEFAULT),
  _mn.efficientnet_b7: lambda: efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT),
  _mn.efficientnet_v2_s: lambda: efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT),
  _mn.efficientnet_v2_m: lambda: efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT),
  _mn.efficientnet_v2_l: lambda: efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT),
  _mn.inception_v3: _download_inception_v3,
  _mn.maxvit_t: lambda: maxvit_t(weights=MaxVit_T_Weights.DEFAULT),
  _mn.mobilenet_v2: lambda: mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT),
  _mn.mobilenet_v3_small: lambda: mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT),
  _mn.mobilenet_v3_large: lambda: mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT),
  _mn.resnet18: lambda: resnet18(weights=ResNet18_Weights.DEFAULT),
  _mn.resnet34: lambda: resnet34(weights=ResNet34_Weights.DEFAULT),
  _mn.resnet50: lambda: resnet50(weights=ResNet50_Weights.DEFAULT),
  _mn.resnet101: lambda: resnet101(weights=ResNet101_Weights.DEFAULT),
  _mn.resnet152: lambda: resnet152(weights=ResNet152_Weights.DEFAULT),
  _mn.swin_t: lambda: swin_t(weights=Swin_T_Weights.DEFAULT),
  _mn.swin_s: lambda: swin_s(weights=Swin_S_Weights.DEFAULT),
  _mn.swin_b: lambda: swin_b(weights=Swin_B_Weights.DEFAULT),
  _mn.swin_v2_t: lambda: swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT),
  _mn.swin_v2_s: lambda: swin_v2_s(weights=Swin_V2_S_Weights.DEFAULT),
  _mn.swin_v2_b: lambda: swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT),
  _mn.vit_b_16: lambda: vit_b_16(weights=ViT_B_16_Weights.DEFAULT),
  _mn.vit_b_32: lambda: vit_b_32(weights=ViT_B_32_Weights.DEFAULT),
  _mn.vit_l_16: lambda: vit_l_16(weights=ViT_L_16_Weights.DEFAULT),
  _mn.vit_l_32: lambda: vit_l_32(weights=ViT_L_32_Weights.DEFAULT),
}


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
    m = PretrainedModelDownloader.download_mobilenet_models(return_model=return_model)
    e = PretrainedModelDownloader.download_efficientnet_models(return_model=return_model)

    if return_model:
      return r + d + m + e

    return []

  @staticmethod
  def download_model(name: str) -> torch.nn.Module:
    fn = _PRETRAINED_LOADERS.get(name)
    if fn is None:
      raise ValueError(f"Model name not found: {name}")
    return fn()


if __name__ == "__main__":
  PretrainedModelDownloader.download_all_models()
