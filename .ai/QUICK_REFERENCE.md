# 🚀 Quick Reference Card

## 🔧 Essential Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Common Imports
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from evaluation.metrics import Metrics
from attacks.attack_factory import AttackFactory
```

## 📊 Key Metrics

### Classification Metrics
```python
# Using scikit-learn (recommended)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')
```

### Distance Metrics
```python
# L1, L2, L∞ distances
l1_dist = torch.norm(img1 - img2, p=1)
l2_dist = torch.norm(img1 - img2, p=2)
linf_dist = torch.norm(img1 - img2, p=float('inf'))
```

## 🎯 Attack Patterns

### Basic Attack Structure
```python
class MyAttack(Attack):
    def __init__(self, model, **kwargs):
        super().__init__(model)
        self.epsilon = kwargs.get('epsilon', 0.1)
    
    @torch.no_grad()
    def attack(self, images, labels):
        # Attack implementation
        return adversarial_images
```

### Common Attack Parameters
```python
# FGSM
attack = AttackFactory.create_attack('fgsm', model, epsilon=0.03)

# PGD
attack = AttackFactory.create_attack('pgd', model, epsilon=0.03, alpha=0.01, steps=40)

# C&W
attack = AttackFactory.create_attack('cw', model, c=1.0, kappa=0, steps=1000)
```

## 🏗️ Model Patterns

### Model Loading
```python
# Load pretrained model
model = torch.load('model.pt', map_location='cuda')
model.eval()

# Load state dict
model = ResNet18(num_classes=10)
model.load_state_dict(torch.load('model_state.pt'))
model.eval()
```

### Model Evaluation
```python
model.eval()
with torch.no_grad():
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = accuracy_score(labels.cpu(), predictions.cpu())
```

## 🔍 Debugging Quick Fixes

### Common Issues
```python
# CUDA out of memory
torch.cuda.empty_cache()
# Reduce batch size

# Device mismatch
tensor = tensor.to(device)
# Check: tensor.device

# Model in wrong mode
model.eval()  # For inference
model.train()  # For training

# Gradient issues
with torch.no_grad():  # For inference
    outputs = model(images)
```

### Memory Monitoring
```python
# Check GPU memory
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Clear cache
torch.cuda.empty_cache()
```

## 📁 File Structure Quick Look

```
sem-vi-code/
├── architectures/     # Model definitions
├── attacks/          # Attack implementations
├── evaluation/       # Metrics and validation
├── training/         # Training scripts
├── data_eng/         # Data utilities
├── domain/           # Core entities
├── shared/           # Shared utilities
├── requirements.txt  # Dependencies
└── SETUP.md         # Setup guide
```

## 🚨 Error Codes & Solutions

| Error | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce batch size, clear cache |
| `Device mismatch` | Use `.to(device)` consistently |
| `Import error` | Check virtual environment activation |
| `Model not in eval mode` | Call `model.eval()` |
| `Gradient computation` | Use `@torch.no_grad()` for inference |

## 🔧 Performance Tips

### GPU Optimization
```python
# Use mixed precision
from torch.cuda.amp import autocast, GradScaler

# Efficient data loading
dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)

# Memory efficient
torch.backends.cudnn.benchmark = True
```

### Batch Processing
```python
# Process in batches
for batch in dataloader:
    images, labels = batch
    # Process batch
    torch.cuda.empty_cache()  # Clear cache if needed
```

## 📊 Dataset Quick Access

### CIFAR-10
```python
# 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
# 32x32 RGB images
# 50,000 train, 10,000 test
```

### ImageNette
```python
# 10 classes: tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute
# Variable size RGB images
# Subset of ImageNet
```

## 🎯 Research Quick Notes

### Attack Success Rate
```python
# Calculate attack success
success_rate = (original_correct & adversarial_incorrect).float().mean()
```

### Transferability
```python
# Test attack transferability
attack_on_model_a = attack.attack(model_a, images)
success_on_model_b = evaluate_attack(model_b, attack_on_model_a, labels)
```

### Robustness Evaluation
```python
# Evaluate model robustness
clean_acc = evaluate_model(model, clean_loader)
adv_acc = evaluate_model(model, adversarial_loader)
robustness = adv_acc / clean_acc
```

---

**💡 Tip**: Keep this file open for quick reference while working on the project!
