# 🤖 AI Assistant Context - Adversarial Attack Research Project

## 📋 Project Summary
This repository contains a comprehensive framework for studying adversarial attacks against deep learning models. It implements various attack methods, evaluation metrics, and training procedures for robust model development.

## 🏗️ Architecture Overview

### Core Components
1. **Neural Network Architectures** (`architectures/`)
   - ResNet (18, 50, 101, 152)
   - VGG (11, 13, 16, 19)
   - DenseNet (121, 161, 169, 201)
   - EfficientNet (B0)
   - MobileNetV2

2. **Adversarial Attacks** (`attacks/`)
   - **White-box attacks**: FGSM, PGD, C&W, DeepFool, etc.
   - **Black-box attacks**: Square, Pixle
   - **Transferability studies**: Cross-model attack effectiveness

3. **Evaluation Framework** (`evaluation/`)
   - Classification metrics (accuracy, precision, recall, F1)
   - Distance metrics (L1, L2, L∞)
   - Attack success rates and robustness measures
   - Visualization tools for attack patterns

4. **Training Pipeline** (`training/`)
   - Standard training procedures
   - Adversarial training implementations
   - Transfer learning capabilities

## 🔬 Research Focus Areas

### 1. Attack Effectiveness
- Success rates across different model architectures
- Transferability between models
- Attack strength vs. perturbation magnitude

### 2. Model Robustness
- Adversarial training effectiveness
- Architecture-specific robustness patterns
- Dataset-specific vulnerabilities

### 3. Evaluation Metrics
- Standard classification metrics
- Adversarial-specific measures
- Distance-based attack strength

## 📊 Datasets Used
- **CIFAR-10**: 32x32 color images, 10 classes
- **ImageNette**: Subset of ImageNet, 10 classes
- **ImageNet**: Large-scale image classification
- **MNIST**: Handwritten digits (28x28 grayscale)

## 🛠️ Technical Stack
- **PyTorch 2.0.0**: Deep learning framework with CUDA support
- **scikit-learn**: Robust metrics implementation
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization
- **tqdm**: Progress tracking

## 🔧 Key Implementation Details

### Metrics System
- Refactored to use scikit-learn for reliability
- Multi-class classification support
- GPU-accelerated distance calculations
- Proper handling of edge cases

### Attack Framework
- Modular design for easy extension
- Consistent interface across attack types
- Support for both targeted and untargeted attacks
- Configurable parameters

### Training Pipeline
- Support for multiple architectures
- Adversarial training capabilities
- Transfer learning support
- Comprehensive logging and checkpointing

## 🎯 Common Use Cases

### 1. Evaluating Model Robustness
```python
from evaluation.metrics import Metrics
from attacks.attack_factory import AttackFactory

# Load model and data
model = load_model()
data_loader = load_data()

# Run attacks
attack = AttackFactory.create_attack('fgsm', model)
results = attack.attack(data_loader)

# Evaluate
metrics = Metrics.evaluate_attack(results, num_classes=10)
```

### 2. Training Robust Models
```python
from training.adversarial_training import AdversarialTrainer

trainer = AdversarialTrainer(model, attack_type='pgd')
trainer.train(train_loader, val_loader, epochs=100)
```

### 3. Attack Transferability Studies
```python
from attacks.transferability import TransferabilityStudy

study = TransferabilityStudy()
results = study.evaluate_transferability(
    source_models=[resnet18, vgg16],
    target_models=[densenet121, efficientnet],
    attack_type='fgsm'
)
```

## 🚨 Important Considerations

### Performance
- Always use `@torch.no_grad()` for inference
- Leverage CUDA when available
- Use batch processing for efficiency
- Monitor GPU memory usage

### Reproducibility
- Set random seeds for consistent results
- Use deterministic algorithms when possible
- Document hyperparameters and configurations
- Version control model checkpoints

### Security
- Be mindful of adversarial attack implications
- Use responsibly for research purposes only
- Follow ethical guidelines for AI research
- Consider real-world attack scenarios

## 📈 Research Directions

### Current Capabilities
- Comprehensive attack implementation
- Robust evaluation framework
- Multiple architecture support
- Transferability analysis

### Future Extensions
- New attack methods
- Advanced defense mechanisms
- Real-time attack detection
- Interpretability studies

## 🔍 Debugging Tips

### Common Issues
1. **CUDA out of memory**: Reduce batch size or use gradient accumulation
2. **Import errors**: Check virtual environment activation
3. **Metric calculation errors**: Verify input tensor shapes and types
4. **Attack failures**: Check model output format and attack parameters

### Debugging Tools
- Use `torch.cuda.is_available()` to check GPU status
- Monitor memory usage with `torch.cuda.memory_allocated()`
- Use `print()` statements for tensor shapes and values
- Check model outputs with `model.eval()` mode

## 📚 References
- Adversarial Attacks and Defenses in Deep Learning
- PyTorch Documentation
- scikit-learn Metrics Guide
- Research papers on adversarial robustness

---

This context file helps AI assistants understand the project structure, research goals, and implementation patterns for better assistance.
