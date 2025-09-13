import torch
import torch.nn as nn

class SetupPretraining:
  @staticmethod
  def setup_imagenette(model: torch.nn.Module, num_last_layers_to_unfreeze: int = 2):
    """
    Setup model for ImageNette transfer learning with proper layer freezing and classifier replacement.
    
    Args:
        model: Pretrained ImageNet model
        num_last_layers_to_unfreeze: Number of last layers to unfreeze (default: 2)
    
    Returns:
        Model configured for ImageNette (10 classes)
    """
    
    # Store original classifier info for debugging
    original_classifier_info = SetupPretraining._get_classifier_info(model)
    print(f"Original classifier: {original_classifier_info}")
    
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze specific layers based on model architecture
    SetupPretraining._unfreeze_layers(model, num_last_layers_to_unfreeze)
    
    # Replace classifier with ImageNette-specific one (10 classes)
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
  def _get_classifier_info(model):
    """Get information about the current classifier layer."""
    if hasattr(model, 'fc'):
        return f"fc: {model.fc}"
    elif hasattr(model, 'classifier'):
        return f"classifier: {model.classifier}"
    elif hasattr(model, '_fc'):
        return f"_fc: {model._fc}"
    else:
        return "No classifier found"
  
  @staticmethod
  def _unfreeze_layers(model, num_layers):
    """Unfreeze the last N layers based on model architecture."""
    model_name = model.__class__.__name__.lower()
    
    if 'resnet' in model_name:
        # For ResNet: unfreeze last few layers
        layers_to_unfreeze = ['layer4', 'fc']
        for name, module in model.named_modules():
            if any(layer in name for layer in layers_to_unfreeze):
                for param in module.parameters():
                    param.requires_grad = True
                    
    elif 'densenet' in model_name:
        # For DenseNet: unfreeze classifier and last dense block
        layers_to_unfreeze = ['classifier', 'features.denseblock4']
        for name, module in model.named_modules():
            if any(layer in name for layer in layers_to_unfreeze):
                for param in module.parameters():
                    param.requires_grad = True
                    
    elif 'vgg' in model_name:
        # For VGG: unfreeze classifier
        for name, module in model.named_modules():
            if 'classifier' in name:
                for param in module.parameters():
                    param.requires_grad = True
                    
    elif 'mobilenet' in model_name:
        # For MobileNet: unfreeze classifier
        for name, module in model.named_modules():
            if 'classifier' in name:
                for param in module.parameters():
                    param.requires_grad = True
                    
    elif 'efficientnet' in model_name:
        # For EfficientNet: unfreeze classifier
        for name, module in model.named_modules():
            if 'classifier' in name:
                for param in module.parameters():
                    param.requires_grad = True
  
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
        
    else:
        raise ValueError("Model type not recognized. Can't replace the classifier.")
  
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

