# 🧠 AI Assistant Context - Adversarial Attack Research

## 🎯 Project Mission
This repository implements a comprehensive framework for studying adversarial attacks against deep learning models, focusing on robustness evaluation and attack transferability.

## 🏗️ Core Architecture

### Key Components
- **Neural Networks**: ResNet, VGG, DenseNet, EfficientNet, MobileNetV2
- **Attacks**: 30+ white-box and black-box attack implementations
- **Evaluation**: Robust metrics using scikit-learn and PyTorch
- **Datasets**: CIFAR-10, ImageNette, ImageNet, MNIST

### File Organization
```
sem-vi-code/
├── architectures/     # Model definitions
├── attacks/          # Attack implementations
│   ├── white_box/   # FGSM, PGD, C&W, DeepFool, etc.
│   └── black_box/   # Square, Pixle
├── evaluation/       # Metrics and validation
├── training/         # Training scripts
├── data_eng/         # Data utilities
├── domain/           # Core entities
└── .ai/             # AI context files
```

## 🔧 Technical Stack

### Primary Dependencies
- **PyTorch 2.0.0** with CUDA support
- **scikit-learn** for robust metrics
- **NumPy** for numerical operations
- **Matplotlib** for visualization

### Key Patterns
- All attacks inherit from base `Attack` class
- Metrics use scikit-learn for reliability
- Models support both CPU and GPU operations
- Comprehensive error handling and validation

## 🎯 Common Tasks

### 1. Implementing New Attacks
```python
class MyAttack(Attack):
    def __init__(self, model, **kwargs):
        super().__init__(model)
        self.epsilon = kwargs.get('epsilon', 0.1)
    
    @torch.no_grad()
    def attack(self, images, labels):
        # Implementation here
        return adversarial_images
```

### 2. Evaluating Model Robustness
```python
from evaluation.metrics import Metrics
from attacks.attack_factory import AttackFactory

# Create attack
attack = AttackFactory.create_attack('fgsm', model, epsilon=0.03)

# Run attack
results = attack.attack(data_loader)

# Evaluate
metrics = Metrics.evaluate_attack(results, num_classes=10)
```

### 3. Training Robust Models
```python
from training.adversarial_training import AdversarialTrainer

trainer = AdversarialTrainer(model, attack_type='pgd')
trainer.train(train_loader, val_loader, epochs=100)
```

## 🚨 Critical Guidelines

### Code Quality
- **Always use type hints** for function parameters and return values
- **Prefer library functions** over custom implementations
- **Use `@torch.no_grad()`** for all inference operations
- **Handle edge cases** (empty tensors, division by zero)

### Performance
- **Check CUDA availability** before GPU operations
- **Use batch processing** for efficiency
- **Monitor memory usage** with `torch.cuda.memory_allocated()`
- **Clear GPU cache** when needed: `torch.cuda.empty_cache()`

### Error Handling
- **Validate inputs** before processing
- **Provide meaningful error messages** with context
- **Handle device mismatches** (CPU vs GPU tensors)
- **Check tensor shapes** before operations

## 🔍 Debugging Patterns

### Common Issues
1. **CUDA out of memory**: Reduce batch size, clear cache
2. **Device mismatch**: Use `.to(device)` consistently
3. **Import errors**: Check virtual environment activation
4. **Model not in eval mode**: Call `model.eval()`
5. **Gradient computation**: Use `@torch.no_grad()` for inference

### Debugging Code
```python
# Check tensor properties
print(f"Shape: {tensor.shape}, Device: {tensor.device}, Dtype: {tensor.dtype}")

# Check model state
print(f"Model mode: {'train' if model.training else 'eval'}")

# Check CUDA status
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

## 📊 Research Context

### Attack Types
- **White-box**: FGSM, PGD, C&W, DeepFool, JSMA, etc.
- **Black-box**: Square, Pixle
- **Transferability**: Cross-model attack effectiveness

### Evaluation Metrics
- **Classification**: Accuracy, Precision, Recall, F1
- **Distance**: L1, L2, L∞ norms
- **Attack Success**: Success rates and robustness measures

### Datasets
- **CIFAR-10**: 32x32 color images, 10 classes
- **ImageNette**: ImageNet subset, 10 classes
- **ImageNet**: Large-scale classification
- **MNIST**: Handwritten digits

## 🎯 AI Assistant Priorities

### When Implementing Features
1. **Follow existing patterns** from similar implementations
2. **Use established libraries** (scikit-learn, PyTorch)
3. **Ensure CUDA compatibility** for GPU operations
4. **Add comprehensive error handling**
5. **Include proper documentation**

### When Debugging Issues
1. **Check tensor shapes and types**
2. **Verify device placement (CPU/GPU)**
3. **Test with small batches first**
4. **Use print statements for intermediate values**
5. **Check model state (train/eval mode)**

### When Providing Suggestions
1. **Prioritize performance and efficiency**
2. **Maintain code consistency**
3. **Consider research implications**
4. **Provide working examples**
5. **Include error handling**

## 🔄 Iteration Process

### Code Review Checklist
- [ ] Type hints present
- [ ] Error handling implemented
- [ ] GPU compatibility checked
- [ ] Documentation added
- [ ] Edge cases handled
- [ ] Performance considerations addressed

### Testing Strategy
1. **Test with small examples first**
2. **Verify on both CPU and GPU**
3. **Check memory usage**
4. **Validate output correctness**
5. **Test edge cases**

---

**Remember**: This is a research project focused on adversarial attacks. Always prioritize code quality, performance, and scientific rigor while maintaining ethical considerations.
