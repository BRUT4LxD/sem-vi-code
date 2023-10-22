import torch
from data_eng.pretrained_model_downloader import PretrainedModelDownloader


class ImageNetModels:
  @staticmethod
  def get_model(model_name: str) -> torch.nn.Module:
    return PretrainedModelDownloader.download_model(model_name)

  @staticmethod
  def get_all_models() -> list:
    return PretrainedModelDownloader.download_all_models(return_model=True)
