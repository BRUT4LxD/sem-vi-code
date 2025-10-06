# Adversarial Noise Detection Training Guide

This guide shows how to train binary classifiers to detect adversarial examples in ImageNette using the newly created training infrastructure.

## Quick Start

### Method 1: Using ImageNetteModelTrainer (Recommended)

```python
from imagenette_lab.imagenette_model_trainer import ImageNetteModelTrainer
from domain.model.model_names import ModelNames

# Initialize trainer
trainer = ImageNetteModelTrainer()

# Train single noise detection model
result = trainer.train_noise_detection_model(
    model_name=ModelNames().resnet18,
    attacked_images_folder='data/attacks/imagenette_models',
    batch_size=32,
    learning_rate=0.001,
    num_epochs=20
)

# Check results
if result['success']:
    print(f"Accuracy: {result['best_val_accuracy']:.2f}%")
    print(f"F1 Score: {result['best_f1']:.2f}%")
    print(f"Saved to: {result['save_path']}")
```

### Method 2: Using Training.train_imagenette_noise_detection Directly

```python
from config.imagenet_models import ImageNetModels
from data_eng.dataset_loader import load_attacked_imagenette
from training.train import Training

# Load dataset
train_loader, test_loader = load_attacked_imagenette(
    attacked_images_folder="data/attacks/imagenette_models",
    clean_train_folder="./data/imagenette/train",
    clean_test_folder="./data/imagenette/val",
    test_images_per_attack=2,
    batch_size=32
)

# Setup and train model
model = ImageNetModels.get_model('resnet18')
training_state = Training.train_imagenette_noise_detection(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    learning_rate=0.001,
    num_epochs=20,
    save_model_path="./models/noise_detection/resnet18_noise_detector.pt"
)
```

## Training Multiple Models

```python
from imagenette_lab.imagenette_model_trainer import ImageNetteModelTrainer
from domain.model.model_names import ModelNames

trainer = ImageNetteModelTrainer()

# Train multiple models at once
results = trainer.train_multiple_noise_detectors(
    model_names=[
        ModelNames().resnet18,
        ModelNames().densenet121,
        ModelNames().mobilenet_v2,
        ModelNames().efficientnet_b0
    ],
    attacked_images_folder='data/attacks/imagenette_models',
    batch_size=32,
    learning_rate=0.001,
    num_epochs=20
)

# Results are sorted by F1 score
for result in results:
    if result['success']:
        print(f"{result['model_name']}: {result['best_f1']:.2f}% F1")
```

## Validating Trained Models

```python
from imagenette_lab.imagenette_model_trainer import ImageNetteModelTrainer

trainer = ImageNetteModelTrainer()

# Validate a trained noise detector
result = trainer.validate_noise_detector(
    model_path='./models/noise_detection/resnet18_noise_detector.pt',
    attacked_images_folder='data/attacks/imagenette_models',
    batch_size=32
)

if result['success']:
    metrics = result['evaluation_metrics']
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Precision: {metrics['precision']:.2f}%")
    print(f"Recall: {metrics['recall']:.2f}%")
    print(f"F1 Score: {metrics['f1']:.2f}%")
```

## Dataset Details

### Data Sources
- **Adversarial Images**: From `data/attacks/imagenette_models/`
  - Generated using `attack_and_save_images_multiple()` from `imagenette_direct_attacks.py`
  - Combines images from **all models** and **all attacks**
  - Skips `src_` prefixed images (clean source images)

- **Clean Training Images**: From `./data/imagenette/train/`
  - Original ImageNette training set
- **Clean Test Images**: From `./data/imagenette/val/`
  - Original ImageNette validation set

### Dataset Splits
- **Training Set**:
  - Adversarial images (label=1): All except `test_images_per_attack` per attack
  - Clean images (label=0): **2x** the number of adversarial images (from train folder)
  - Ratio: 2:1 (clean:adversarial)

- **Test Set**:
  - Adversarial images (label=1): `test_images_per_attack` per attack
  - Clean images (label=0): Equal to adversarial images (from val folder)
  - Ratio: **50:50** (balanced)

### Labels
- **0 = Clean Image** (not attacked)
- **1 = Adversarial Image** (attacked)

## Training Parameters

### Default Parameters
```python
attacked_images_folder = "data/attacks/imagenette_models"
clean_train_folder = "./data/imagenette/train"
clean_test_folder = "./data/imagenette/val"
test_images_per_attack = 2
batch_size = 32
learning_rate = 0.001
num_epochs = 20
early_stopping_patience = 7
scheduler_type = 'plateau'
weight_decay = 0.0001
gradient_clip_norm = 1.0
```

### Recommended Configurations

#### Fast Training (Testing)
```python
result = trainer.train_noise_detection_model(
    model_name='resnet18',
    batch_size=64,
    learning_rate=0.01,
    num_epochs=5,
    early_stopping_patience=3
)
```

#### Balanced Training (Recommended)
```python
result = trainer.train_noise_detection_model(
    model_name='resnet18',
    batch_size=32,
    learning_rate=0.001,
    num_epochs=20,
    early_stopping_patience=7
)
```

#### High Accuracy Training
```python
result = trainer.train_noise_detection_model(
    model_name='efficientnet_b0',
    batch_size=16,
    learning_rate=0.0005,
    num_epochs=30,
    early_stopping_patience=10,
    weight_decay=0.001,
    gradient_clip_norm=0.5
)
```

## Output Structure

### Saved Models
Models are saved to: `./models/noise_detection/{model_name}_noise_detector.pt`

### Checkpoint Contents
```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': 15,
    'val_accuracy': 95.5,
    'val_loss': 0.123,
    'precision': 94.2,
    'recall': 96.8,
    'f1': 95.5,
    'training_state': {
        'best_val_accuracy': 95.5,
        'best_epoch': 12,
        'best_precision': 94.2,
        'best_recall': 96.8,
        'best_f1': 95.5,
        'train_losses': [...],
        'val_accuracies': [...],
        'total_training_time': 1234.5,
        ...
    }
}
```

### Training Results Dictionary
```python
result = {
    'model_name': 'resnet18',
    'task': 'noise_detection',
    'save_path': './models/noise_detection/resnet18_noise_detector.pt',
    'best_val_accuracy': 95.5,
    'best_precision': 94.2,
    'best_recall': 96.8,
    'best_f1': 95.5,
    'training_time': 1234.5,
    'training_results': {...},  # Complete training history
    'success': True
}
```

## Metrics Explained

### Confusion Matrix
- **True Positives (TP)**: Adversarial images correctly detected
- **False Positives (FP)**: Clean images incorrectly flagged as adversarial
- **True Negatives (TN)**: Clean images correctly identified
- **False Negatives (FN)**: Adversarial images missed

### Key Metrics
- **Accuracy**: Overall correctness = (TP + TN) / Total
- **Precision**: When we predict adversarial, how often correct = TP / (TP + FP)
- **Recall**: Of all adversarial images, how many detected = TP / (TP + FN)
- **F1 Score**: Harmonic mean of precision and recall = 2 * (P * R) / (P + R)

### Why F1 Matters
For noise detection, F1 is often more important than accuracy because:
- High precision: Minimizes false alarms
- High recall: Catches most adversarial examples
- F1 balances both concerns

## Complete Workflow Example

```python
from imagenette_lab.imagenette_model_trainer import ImageNetteModelTrainer
from domain.model.model_names import ModelNames

# Step 1: Initialize trainer
trainer = ImageNetteModelTrainer()

# Step 2: Train noise detector
print("Training noise detection model...")
training_result = trainer.train_noise_detection_model(
    model_name=ModelNames().resnet18,
    attacked_images_folder='data/attacks/imagenette_models',
    clean_train_folder='./data/imagenette/train',
    clean_test_folder='./data/imagenette/val',
    test_images_per_attack=2,
    batch_size=32,
    learning_rate=0.001,
    num_epochs=20,
    early_stopping_patience=7,
    scheduler_type='plateau',
    weight_decay=0.0001,
    gradient_clip_norm=1.0,
    verbose=True
)

# Step 3: Check results
if training_result['success']:
    print(f"\n✅ Training successful!")
    print(f"   Model: {training_result['model_name']}")
    print(f"   Accuracy: {training_result['best_val_accuracy']:.2f}%")
    print(f"   F1 Score: {training_result['best_f1']:.2f}%")
    print(f"   Saved to: {training_result['save_path']}")
    
    # Step 4: Validate the trained model
    validation_result = trainer.validate_noise_detector(
        model_path=training_result['save_path'],
        attacked_images_folder='data/attacks/imagenette_models',
        batch_size=32
    )
    
    if validation_result['success']:
        metrics = validation_result['evaluation_metrics']
        print(f"\n📊 Validation Metrics:")
        print(f"   Accuracy: {metrics['accuracy']:.2f}%")
        print(f"   Precision: {metrics['precision']:.2f}%")
        print(f"   Recall: {metrics['recall']:.2f}%")
        print(f"   F1 Score: {metrics['f1']:.2f}%")
else:
    print(f"❌ Training failed: {training_result['error']}")
```

## TensorBoard Monitoring

Training automatically logs to TensorBoard:

```bash
# View training progress
tensorboard --logdir=runs/binary_training
```

**Logged Metrics:**
- Train/Validation Loss
- Train/Validation Accuracy
- Precision, Recall, F1 Score
- Learning Rate
- Epoch Time

## File Structure

```
models/
└── noise_detection/          ← New folder for noise detectors
    ├── resnet18_noise_detector.pt
    ├── densenet121_noise_detector.pt
    ├── mobilenet_v2_noise_detector.pt
    └── efficientnet_b0_noise_detector.pt

runs/
└── binary_training/          ← TensorBoard logs
    ├── resnet18/
    ├── densenet121/
    └── ...
```

## Troubleshooting

### Issue: Not enough clean images
**Solution**: Reduce the number of adversarial images or use a different clean dataset

### Issue: Low accuracy
**Solutions**:
- Increase number of epochs
- Adjust learning rate (try 0.0001 or 0.01)
- Increase weight_decay for regularization
- Use gradient clipping (gradient_clip_norm=1.0)

### Issue: Overfitting
**Solutions**:
- Increase weight_decay (try 0.001)
- Reduce num_epochs
- Increase early_stopping_patience
- Use data augmentation in transforms

### Issue: Training too slow
**Solutions**:
- Increase batch_size (if GPU memory allows)
- Reduce number of models/attacks in dataset
- Use fewer epochs for initial testing

## Advanced Usage

### Custom Transform
```python
from torchvision import transforms

custom_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_loader, test_loader = load_attacked_imagenette(
    transform=custom_transform
)
```

### Using Trained Model for Inference
```python
from data_eng.io import load_model_binary
import torch
from PIL import Image
from torchvision import transforms

# Load trained model
result = load_model_binary('./models/noise_detection/resnet18_noise_detector.pt')
model = result['model']
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open('test_image.png')
image_tensor = transform(image).unsqueeze(0).to('cuda')

# Predict
with torch.no_grad():
    output = model(image_tensor)
    probability = torch.sigmoid(output).item()
    is_adversarial = probability > 0.5

print(f"Adversarial probability: {probability:.2%}")
print(f"Is adversarial: {is_adversarial}")
```

## Expected Results

Based on typical adversarial detection tasks:
- **Accuracy**: 85-95%
- **Precision**: 80-95%
- **Recall**: 85-98%
- **F1 Score**: 85-95%

Higher scores indicate better adversarial detection capability.

## Summary

**Complete Workflow:**
1. Generate adversarial images → `attack_and_save_images_multiple()`
2. Load dataset → `load_attacked_imagenette()`
3. Train detector → `trainer.train_noise_detection_model()`
4. Validate → `trainer.validate_noise_detector()`
5. Deploy → Use trained model for inference

Your noise detection system is ready for production! 🎯

