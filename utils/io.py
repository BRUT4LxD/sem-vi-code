import torch


def save(model, model_save_path):
    torch.save(model, model_save_path)


def load(model_load_path):
    try:
      model = torch.load(model_load_path)
    except:
      return None

    return model
