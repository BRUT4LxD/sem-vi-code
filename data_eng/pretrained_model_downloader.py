from __future__ import annotations

import torch
import torchvision.models as tvm
from typing import Callable, Dict, List, Tuple

from domain.model.model_names import ModelNames


def _loader(ctor_name: str, weights_name: str):
  ctor = getattr(tvm, ctor_name)
  wcls = getattr(tvm, weights_name)

  def _fn() -> torch.nn.Module:
    return ctor(weights=wcls.DEFAULT)

  return _fn


# (ModelNames key, torchvision factory name, torchvision *Weights class name)
_TORCHVISION_DEFAULT_SPECS: Tuple[Tuple[str, str, str], ...] = (
  (ModelNames.densenet121, "densenet121", "DenseNet121_Weights"),
  (ModelNames.densenet161, "densenet161", "DenseNet161_Weights"),
  (ModelNames.densenet169, "densenet169", "DenseNet169_Weights"),
  (ModelNames.densenet201, "densenet201", "DenseNet201_Weights"),
  (ModelNames.efficientnet_b0, "efficientnet_b0", "EfficientNet_B0_Weights"),
  (ModelNames.efficientnet_b1, "efficientnet_b1", "EfficientNet_B1_Weights"),
  (ModelNames.efficientnet_b2, "efficientnet_b2", "EfficientNet_B2_Weights"),
  (ModelNames.efficientnet_b3, "efficientnet_b3", "EfficientNet_B3_Weights"),
  (ModelNames.efficientnet_b4, "efficientnet_b4", "EfficientNet_B4_Weights"),
  (ModelNames.efficientnet_b5, "efficientnet_b5", "EfficientNet_B5_Weights"),
  (ModelNames.efficientnet_b6, "efficientnet_b6", "EfficientNet_B6_Weights"),
  (ModelNames.efficientnet_b7, "efficientnet_b7", "EfficientNet_B7_Weights"),
  (ModelNames.efficientnet_v2_s, "efficientnet_v2_s", "EfficientNet_V2_S_Weights"),
  (ModelNames.efficientnet_v2_m, "efficientnet_v2_m", "EfficientNet_V2_M_Weights"),
  (ModelNames.efficientnet_v2_l, "efficientnet_v2_l", "EfficientNet_V2_L_Weights"),
  (ModelNames.inception_v3, "inception_v3", "Inception_V3_Weights"),
  (ModelNames.maxvit_t, "maxvit_t", "MaxVit_T_Weights"),
  (ModelNames.mobilenet_v2, "mobilenet_v2", "MobileNet_V2_Weights"),
  (ModelNames.mobilenet_v3_small, "mobilenet_v3_small", "MobileNet_V3_Small_Weights"),
  (ModelNames.mobilenet_v3_large, "mobilenet_v3_large", "MobileNet_V3_Large_Weights"),
  (ModelNames.resnet18, "resnet18", "ResNet18_Weights"),
  (ModelNames.resnet34, "resnet34", "ResNet34_Weights"),
  (ModelNames.resnet50, "resnet50", "ResNet50_Weights"),
  (ModelNames.resnet101, "resnet101", "ResNet101_Weights"),
  (ModelNames.resnet152, "resnet152", "ResNet152_Weights"),
  (ModelNames.swin_t, "swin_t", "Swin_T_Weights"),
  (ModelNames.swin_s, "swin_s", "Swin_S_Weights"),
  (ModelNames.swin_b, "swin_b", "Swin_B_Weights"),
  (ModelNames.swin_v2_t, "swin_v2_t", "Swin_V2_T_Weights"),
  (ModelNames.swin_v2_s, "swin_v2_s", "Swin_V2_S_Weights"),
  (ModelNames.swin_v2_b, "swin_v2_b", "Swin_V2_B_Weights"),
  (ModelNames.vit_b_16, "vit_b_16", "ViT_B_16_Weights"),
  (ModelNames.vit_b_32, "vit_b_32", "ViT_B_32_Weights"),
  (ModelNames.vit_l_16, "vit_l_16", "ViT_L_16_Weights"),
  (ModelNames.vit_l_32, "vit_l_32", "ViT_L_32_Weights"),
)

_PRETRAINED_LOADERS: Dict[str, Callable[[], torch.nn.Module]] = {
  key: _loader(ctor, w) for key, ctor, w in _TORCHVISION_DEFAULT_SPECS
}

_PRETRAINED_KEYS_IN_SPEC_ORDER: Tuple[str, ...] = tuple(k for k, _, _ in _TORCHVISION_DEFAULT_SPECS)


class PretrainedModelDownloader:

  @staticmethod
  def download_resnet_models(return_model: bool = False) -> list:
    r1 = _PRETRAINED_LOADERS[ModelNames.resnet18]()
    r2 = _PRETRAINED_LOADERS[ModelNames.resnet50]()
    r3 = _PRETRAINED_LOADERS[ModelNames.resnet101]()
    r4 = _PRETRAINED_LOADERS[ModelNames.resnet152]()

    if return_model:
      return [r1, r2, r3, r4]

    return []

  @staticmethod
  def download_densenet_models(return_model: bool = False) -> list:
    d1 = _PRETRAINED_LOADERS[ModelNames.densenet121]()
    d2 = _PRETRAINED_LOADERS[ModelNames.densenet161]()
    d3 = _PRETRAINED_LOADERS[ModelNames.densenet169]()
    d4 = _PRETRAINED_LOADERS[ModelNames.densenet201]()

    if return_model:
      return [d1, d2, d3, d4]

    return []

  @staticmethod
  def download_mobilenet_models(return_model: bool = False):
    m1 = _PRETRAINED_LOADERS[ModelNames.mobilenet_v2]()
    m2 = _PRETRAINED_LOADERS[ModelNames.mobilenet_v3_small]()
    m3 = _PRETRAINED_LOADERS[ModelNames.mobilenet_v3_large]()

    if return_model:
      return [m1, m2, m3]

    return []

  @staticmethod
  def download_efficientnet_models(return_model: bool = False):
    e0 = _PRETRAINED_LOADERS[ModelNames.efficientnet_b0]()
    e1 = _PRETRAINED_LOADERS[ModelNames.efficientnet_b1]()
    e2 = _PRETRAINED_LOADERS[ModelNames.efficientnet_b2]()
    e3 = _PRETRAINED_LOADERS[ModelNames.efficientnet_b3]()
    e4 = _PRETRAINED_LOADERS[ModelNames.efficientnet_b4]()
    e5 = _PRETRAINED_LOADERS[ModelNames.efficientnet_b5]()
    e6 = _PRETRAINED_LOADERS[ModelNames.efficientnet_b6]()
    e7 = _PRETRAINED_LOADERS[ModelNames.efficientnet_b7]()

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
