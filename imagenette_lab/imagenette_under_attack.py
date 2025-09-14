from attacks.system_under_attack import SystemUnderAttack
from attacks.white_box.fgsm import FGSM
from config.imagenette_classes import ImageNetteClasses
from data_eng.dataset_loader import load_imagenette
from data_eng.io import load_model_imagenette

# Load your trained ResNet18 model
model_path = "./models/imagenette/resnet18_advanced.pt"
result = load_model_imagenette(model_path, device='cuda')

if not result['success']:
    print(f"❌ Failed to load model: {result['error']}")
    exit()

model = result['model']
print(f"✅ Loaded model: {result['model_name']}")

# Create FGSM attack
attack = FGSM(model)

# Get test data (small subset for quick testing)
_, test_loader = load_imagenette(batch_size=1, test_subset_size=-1)

# Run attack
SystemUnderAttack.simple_attack(attack, test_loader, num_classes=len(ImageNetteClasses.get_classes()), visualize=True)