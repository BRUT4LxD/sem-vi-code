import torch

class SetupPretraining:
  @staticmethod
  def setup_imagenette(model: torch.nn.Module):
    # make all layers trainable
    for param in model.parameters():
      param.requires_grad = True

      # Check the type of model and replace the last layer accordingly
      if hasattr(model, 'fc'):
          # For ResNet-like models
          print('fc')
          model.fc = torch.nn.Linear(model.fc.in_features, 10)
      elif hasattr(model, 'classifier') and isinstance(model.classifier, torch.nn.modules.linear.Linear):
          # For models where classifier is a linear layer
          model.classifier = torch.nn.Linear(model.classifier.in_features, 10)
      elif hasattr(model, 'classifier') and isinstance(model.classifier, torch.nn.Sequential):
          # For models where classifier is a Sequential container with a Linear layer inside
          linear_layer = model.classifier[1]  # accessing the Linear layer
          model.classifier[1] = torch.nn.Linear(linear_layer.in_features, 10)
      elif hasattr(model, '_fc'):
          # For EfficientNet
          print('_fc')
          model._fc = torch.nn.Linear(model._fc.in_features, 10)
      else:
          raise ValueError("Model type not recognized. Can't replace the last layer.")


    return model

