import torch
from data_eng.pretrained_model_downloader import PretrainedModelDownloader


class ImageNetModels:
  @staticmethod
  def get_model(model_name: str) -> torch.nn.Module:
    return PretrainedModelDownloader.download_model(model_name)

  @staticmethod
  def get_all_models() -> list:
    return PretrainedModelDownloader.download_all_models(return_model=True)

  @staticmethod
  def get_all_vgg_models() -> list:
    return PretrainedModelDownloader.download_vgg_models(return_model=True)

  @staticmethod
  def get_all_resnet_models() -> list:
    return PretrainedModelDownloader.download_resnet_models(return_model=True)

  @staticmethod
  def get_all_densenet_models() -> list:
    return PretrainedModelDownloader.download_densenet_models(return_model=True)

  @staticmethod
  def get_all_mobilenet_models() -> list:
    return PretrainedModelDownloader.download_mobilenet_models(return_model=True)

  @staticmethod
  def get_all_efficientnet_models() -> list:
    return PretrainedModelDownloader.download_efficientnet_models(return_model=True)

