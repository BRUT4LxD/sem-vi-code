from torch.nn import Module

from data_eng.io import load_model


class ModelConfig():
    def __init__(self, model: Module, model_save_path: str, load_pretrained: bool):
        if load_pretrained:
            self.model = load_model(model, model_save_path)
            print('Loaded pretrained model from: ' + model_save_path)
        else:
            self.model = model
        self.model_name = self.model.__call__.__name__
        self.model_save_path = model_save_path
        self.load_pretrained = load_pretrained
