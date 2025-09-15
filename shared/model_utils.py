import torch;
class ModelUtils:

    @staticmethod
    def get_model_num_classes(model: torch.nn.Module):
        last_layer = list(model.children())[-1]
        last_layer_outputs = list(last_layer.parameters())[-1]
        return len(last_layer_outputs)

    @staticmethod
    def remove_missclassified(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor, device: str, labels_mapper: dict = None) -> torch.Tensor:
        _ = model.eval()
        _ = model.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)

        if labels_mapper is not None:
            for original, mapped in labels_mapper.items():
                mask = predictions == original
                predictions[mask] = mapped

        images = images[predictions == labels].clone().to(device)
        labels = labels[predictions == labels].clone().to(device)

        return images, labels

    @staticmethod
    def remove_missclassified_imagenette(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        return images[predictions == labels], labels[predictions == labels]