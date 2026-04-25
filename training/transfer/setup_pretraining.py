import torch
import torch.nn as nn

from training.architecture_freeze_policy import ArchitectureFreezePolicy


class SetupPretraining:
  @staticmethod
  def setup_imagenette(
      model: torch.nn.Module,
      num_last_layers_to_unfreeze: int = 2,
      full_finetune: bool = False,
  ):
    """
    Setup model for ImageNette transfer learning with classifier replacement.

    Args:
        model: Pretrained ImageNet model
        num_last_layers_to_unfreeze: Passed to partial-unfreeze policy (reserved / future use)
        full_finetune: If True, replace classifier then unfreeze **all** parameters (full
            fine-tuning). If False, freeze all, partially unfreeze tail, then replace head.
    """
    
    # Store original classifier info for debugging
    original_classifier_info = SetupPretraining._get_classifier_info(model)
    print(f"Original classifier: {original_classifier_info}")
    
    ArchitectureFreezePolicy.freeze_all(model)

    if full_finetune:
      print("⚙️ Full fine-tuning: classifier will be adapted, then all layers unfrozen.")
      SetupPretraining._replace_classifier(model)
      ArchitectureFreezePolicy.unfreeze_all(model)
    else:
      ArchitectureFreezePolicy.apply_partial_transfer_unfreeze(
          model, num_last_layers_to_unfreeze
      )
      SetupPretraining._replace_classifier(model)
    
    # Verify the setup
    new_classifier_info = SetupPretraining._get_classifier_info(model)
    print(f"New classifier: {new_classifier_info}")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.1f}%)")
    
    return model
  
  @staticmethod
  def setup_binary(
      model: torch.nn.Module,
      num_last_layers_to_unfreeze: int = 2,
      full_finetune: bool = False,
  ):
    """
    Setup model for binary classification with classifier replacement.

    Args:
        model: Pretrained ImageNet model
        num_last_layers_to_unfreeze: Passed to partial-unfreeze policy (reserved / future use)
        full_finetune: If True, replace binary head then unfreeze all parameters.
    """
    
    # Store original classifier info for debugging
    original_classifier_info = SetupPretraining._get_classifier_info(model)
    print(f"Original classifier: {original_classifier_info}")
    
    ArchitectureFreezePolicy.freeze_all(model)

    if full_finetune:
      print("⚙️ Full fine-tuning (binary): head replaced, then all layers unfrozen.")
      SetupPretraining._replace_classifier_binary(model)
      ArchitectureFreezePolicy.unfreeze_all(model)
    else:
      ArchitectureFreezePolicy.apply_partial_transfer_unfreeze(
          model, num_last_layers_to_unfreeze
      )
      SetupPretraining._replace_classifier_binary(model)
    
    # Verify the setup
    new_classifier_info = SetupPretraining._get_classifier_info(model)
    print(f"New classifier: {new_classifier_info}")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.1f}%)")
    
    return model
  
  @staticmethod
  def _get_classifier_info(model):
    """Get information about the current classifier layer."""
    if hasattr(model, 'fc'):
        return f"fc: {model.fc}"
    elif hasattr(model, 'classifier'):
        return f"classifier: {model.classifier}"
    elif hasattr(model, '_fc'):
        return f"_fc: {model._fc}"
    elif hasattr(model, 'head'):
        return f"head: {model.head}"
    else:
        return "No classifier found"
  
  @staticmethod
  def _replace_classifier(model):
    """Replace the classifier layer with ImageNette-specific one (10 classes)."""
    device = next(model.parameters()).device
    
    if hasattr(model, 'fc'):
        # ResNet models
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 10).to(device)
        print(f"Replaced fc layer: {in_features} -> 10")
        
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Linear):
            # Single linear layer (some models)
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, 10).to(device)
            print(f"Replaced classifier (Linear): {in_features} -> 10")
            
        elif isinstance(model.classifier, nn.Sequential):
            # Sequential classifier (VGG, DenseNet, MobileNet)
            # Find the last linear layer
            last_linear_idx = -1
            for i, layer in enumerate(model.classifier):
                if isinstance(layer, nn.Linear):
                    last_linear_idx = i
            
            if last_linear_idx != -1:
                in_features = model.classifier[last_linear_idx].in_features
                model.classifier[last_linear_idx] = nn.Linear(in_features, 10).to(device)
                print(f"Replaced classifier[{last_linear_idx}] (Sequential): {in_features} -> 10")
            else:
                raise ValueError("No linear layer found in classifier")
        else:
            raise ValueError(f"Unknown classifier type: {type(model.classifier)}")
            
    elif hasattr(model, '_fc'):
        # EfficientNet models
        in_features = model._fc.in_features
        model._fc = nn.Linear(in_features, 10).to(device)
        print(f"Replaced _fc layer: {in_features} -> 10")

    elif hasattr(model, 'head'):
        # Swin, ViT (torchvision) and other backbones with a `head` module
        if isinstance(model.head, nn.Linear):
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, 10).to(device)
            print(f"Replaced head (Linear): {in_features} -> 10")
        elif isinstance(model.head, nn.Sequential):
            last_linear_idx = -1
            for i, layer in enumerate(model.head):
                if isinstance(layer, nn.Linear):
                    last_linear_idx = i
            if last_linear_idx == -1:
                raise ValueError("No linear layer found in model.head (Sequential)")
            in_features = model.head[last_linear_idx].in_features
            model.head[last_linear_idx] = nn.Linear(in_features, 10).to(device)
            print(
                f"Replaced head[{last_linear_idx}] (Sequential): {in_features} -> 10"
            )
        else:
            raise ValueError(
                f"Model.head type not supported for ImageNette: {type(model.head)}"
            )
        
    else:
        raise ValueError("Model type not recognized. Can't replace the classifier.")

    SetupPretraining._disable_inception_aux_logits(model)
  
  @staticmethod
  def _replace_classifier_binary(model):
    """Replace the classifier layer with binary classification-specific one (1 output)."""
    device = next(model.parameters()).device
    
    if hasattr(model, 'fc'):
        # ResNet models
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1).to(device)
        print(f"Replaced fc layer for binary classification: {in_features} -> 1")
        
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Linear):
            # Single linear layer (some models)
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, 1).to(device)
            print(f"Replaced classifier (Linear) for binary classification: {in_features} -> 1")
            
        elif isinstance(model.classifier, nn.Sequential):
            # Sequential classifier (VGG, DenseNet, MobileNet)
            # Find the last linear layer
            last_linear_idx = -1
            for i, layer in enumerate(model.classifier):
                if isinstance(layer, nn.Linear):
                    last_linear_idx = i
            
            if last_linear_idx != -1:
                in_features = model.classifier[last_linear_idx].in_features
                model.classifier[last_linear_idx] = nn.Linear(in_features, 1).to(device)
                print(f"Replaced classifier[{last_linear_idx}] (Sequential) for binary classification: {in_features} -> 1")
            else:
                raise ValueError("No linear layer found in classifier")
        else:
            raise ValueError(f"Unknown classifier type: {type(model.classifier)}")
            
    elif hasattr(model, '_fc'):
        # EfficientNet models
        in_features = model._fc.in_features
        model._fc = nn.Linear(in_features, 1).to(device)
        print(f"Replaced _fc layer for binary classification: {in_features} -> 1")

    elif hasattr(model, 'head'):
        if isinstance(model.head, nn.Linear):
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, 1).to(device)
            print(f"Replaced head (Linear) for binary: {in_features} -> 1")
        elif isinstance(model.head, nn.Sequential):
            last_linear_idx = -1
            for i, layer in enumerate(model.head):
                if isinstance(layer, nn.Linear):
                    last_linear_idx = i
            if last_linear_idx == -1:
                raise ValueError("No linear layer found in model.head (Sequential)")
            in_features = model.head[last_linear_idx].in_features
            model.head[last_linear_idx] = nn.Linear(in_features, 1).to(device)
            print(
                f"Replaced head[{last_linear_idx}] (Sequential) for binary: {in_features} -> 1"
            )
        else:
            raise ValueError(
                f"Model.head type not supported for binary: {type(model.head)}"
            )
        
    else:
        raise ValueError("Model type not recognized. Can't replace the classifier.")

    SetupPretraining._disable_inception_aux_logits(model)

  @staticmethod
  def _disable_inception_aux_logits(model):
    """
    Disable Inception's auxiliary classifier for ImageNette's 224px pipeline.

    Torchvision Inception can train with an AuxLogits branch, but with 224x224
    inputs that branch can shrink to 3x3 before a 5x5 conv. The main classifier
    works, so remove the aux path for this project's standard ImageNette setup.
    """
    if model.__class__.__name__.lower().startswith("inception"):
        if hasattr(model, "aux_logits"):
            model.aux_logits = False
        if hasattr(model, "AuxLogits"):
            model.AuxLogits = None
  
  @staticmethod
  def verify_model_setup(model, expected_classes=10):
    """Verify that the model is properly set up for ImageNette."""
    # Get device from model
    device = next(model.parameters()).device
    
    # Test forward pass with dummy input on same device as model
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
            actual_classes = output.shape[1]
            
        if actual_classes == expected_classes:
            print(f"✅ Model verification passed: {actual_classes} classes")
            return True
        else:
            print(f"❌ Model verification failed: expected {expected_classes}, got {actual_classes}")
            return False
            
    except Exception as e:
        print(f"❌ Model verification failed with error: {e}")
        return False
