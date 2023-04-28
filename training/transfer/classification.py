from torchvision.models import resnet50, ResNet50_Weights

resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
resnet50(weights=ResNet50_Weights.DEFAULT)
resnet50(weights="IMAGENET1K_V2")
resnet50(weights=None)