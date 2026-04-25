"""
Centralized freezing / unfreezing for torchvision-style backbones used in this project.

Used by :class:`training.transfer.setup_pretraining.SetupPretraining` for transfer learning
(partial head + last blocks) vs full fine-tuning (all parameters trainable).
"""

from __future__ import annotations

import torch.nn as nn


class ArchitectureFreezePolicy:
    """Freeze or unfreeze parameters by architecture family (ResNet, DenseNet, VGG, …)."""

    @staticmethod
    def freeze_all(model: nn.Module) -> None:
        for param in model.parameters():
            param.requires_grad = False

    @staticmethod
    def unfreeze_all(model: nn.Module) -> None:
        for param in model.parameters():
            param.requires_grad = True

    @staticmethod
    def apply_partial_transfer_unfreeze(
        model: nn.Module, num_last_layers_to_unfreeze: int = 2
    ) -> None:
        """
        Unfreeze a small tail of the network + classifier path (typical transfer learning).

        Note:
            ``num_last_layers_to_unfreeze`` is reserved for future finer control; selection
            is currently by fixed module-name patterns per family (legacy behaviour).
        """
        _ = num_last_layers_to_unfreeze  # reserved
        model_name = model.__class__.__name__.lower()

        if "resnet" in model_name:
            layers_to_unfreeze = ("layer4", "fc")
            for name, module in model.named_modules():
                if any(layer in name for layer in layers_to_unfreeze):
                    for param in module.parameters():
                        param.requires_grad = True

        elif "densenet" in model_name:
            layers_to_unfreeze = ("classifier", "features.denseblock4")
            for name, module in model.named_modules():
                if any(layer in name for layer in layers_to_unfreeze):
                    for param in module.parameters():
                        param.requires_grad = True

        elif "vgg" in model_name:
            for name, module in model.named_modules():
                if "classifier" in name:
                    for param in module.parameters():
                        param.requires_grad = True

        elif "mobilenet" in model_name:
            for name, module in model.named_modules():
                if "classifier" in name:
                    for param in module.parameters():
                        param.requires_grad = True

        elif "efficientnet" in model_name:
            for name, module in model.named_modules():
                if "classifier" in name:
                    for param in module.parameters():
                        param.requires_grad = True

        elif "inception" in model_name:
            for name, module in model.named_modules():
                if "fc" in name or "Mixed_7" in name:
                    for param in module.parameters():
                        param.requires_grad = True

        elif "swin" in model_name:
            # Last feature stage + norm before head + head (name patterns match torchvision)
            for name, module in model.named_modules():
                if "head" in name or name == "norm" or name.startswith("features.3"):
                    for param in module.parameters():
                        param.requires_grad = True
