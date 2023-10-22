import torch

def save_model(model: torch.nn.Module, model_save_path: str):
    torch.save(model.state_dict(), model_save_path)


def load_model(model_instance: torch.nn.Module, model_load_path: str) -> torch.nn.Module:
    model_state_dict = torch.load(model_load_path)
    model_instance.load_state_dict(model_state_dict)
    return model_instance

