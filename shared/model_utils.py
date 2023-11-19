import torch;
class ModelUtils:

    @staticmethod
    def get_model_num_classes(model: torch.nn.Module):
        last_layer = list(model.children())[-1]
        last_layer_outputs = list(last_layer.parameters())[-1]
        return len(last_layer_outputs)

    @staticmethod
    def remove_missclassified(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor, device: str) -> torch.Tensor:
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        images = images[predictions == labels].clone().to(device)
        labels = labels[predictions == labels].clone().to(device)
        return images, labels