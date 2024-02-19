from data_eng.pretrained_model_downloader import PretrainedModelDownloader
from torch.nn import Module, Linear

from domain.model_names import ModelNames

class BinaryModels:

  @staticmethod
  def resnet18() -> Module:
    resnet18 = PretrainedModelDownloader.download_model(ModelNames().resnet18)

    num_features = resnet18.fc.in_features
    resnet18.fc = Linear(num_features, 1)

    return resnet18