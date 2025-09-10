# 🤖 AI Assistant Instructions

## 🎯 Primary Objectives
When working with this adversarial attack research project, prioritize:

1. **Code Quality**: Use established libraries over custom implementations
2. **Performance**: Leverage GPU acceleration and efficient algorithms
3. **Reproducibility**: Ensure consistent and deterministic results
4. **Documentation**: Provide clear explanations and examples
5. **Research Focus**: Maintain scientific rigor and ethical considerations

## 🔧 Technical Guidelines

### Code Implementation
- **Always use type hints** for function parameters and return values
- **Prefer library functions** over custom implementations (scikit-learn, PyTorch)
- **Use `@torch.no_grad()`** for all inference operations
- **Handle edge cases** (empty tensors, division by zero, etc.)
- **Add comprehensive docstrings** for all functions and classes

### Performance Optimization
- **Check CUDA availability** before GPU operations
- **Use batch processing** for efficiency
- **Monitor memory usage** with `torch.cuda.memory_allocated()`
- **Clear GPU cache** when needed: `torch.cuda.empty_cache()`
- **Use appropriate data types** (float32 for most operations)

### Error Handling
- **Validate inputs** before processing
- **Provide meaningful error messages** with context
- **Handle device mismatches** (CPU vs GPU tensors)
- **Check tensor shapes** before operations
- **Use try-catch blocks** for risky operations

## 📊 Project-Specific Patterns

### Metrics Implementation
```python
# ✅ Good: Use scikit-learn
from sklearn.metrics import accuracy_score, precision_score
accuracy = accuracy_score(y_true, y_pred)

# ❌ Avoid: Custom implementations unless necessary
# def custom_accuracy(y_true, y_pred): ...
```

### Attack Implementation
```python
# ✅ Good: Follow established patterns
class MyAttack(Attack):
    def __init__(self, model, **kwargs):
        super().__init__(model)
        self.epsilon = kwargs.get('epsilon', 0.1)
    
    @torch.no_grad()
    def attack(self, images, labels):
        # Implementation here
        pass
```

### Model Evaluation
```python
# ✅ Good: Use proper evaluation pattern
model.eval()
with torch.no_grad():
    outputs = model(images)
    predictions = torch.argmax(outputs, dim=1)
    accuracy = accuracy_score(labels.cpu(), predictions.cpu())
```

## 🚨 Common Pitfalls to Avoid

### 1. Memory Issues
- Don't accumulate gradients during evaluation
- Clear intermediate tensors when possible
- Use appropriate batch sizes for available memory

### 2. Device Mismatches
- Always check tensor device before operations
- Use `.to(device)` consistently
- Handle CPU/GPU tensor conversions properly

### 3. Metric Calculations
- Don't use binary classification metrics for multi-class
- Handle zero-division cases properly
- Use appropriate averaging strategies

### 4. Attack Implementation
- Validate attack parameters before use
- Handle edge cases (empty batches, single samples)
- Ensure attack constraints are properly enforced

## 🔍 Debugging Strategies

### When Issues Arise
1. **Check tensor shapes and types**
2. **Verify device placement (CPU/GPU)**
3. **Test with small batches first**
4. **Use print statements for intermediate values**
5. **Check model state (train/eval mode)**

### Useful Debugging Code
```python
# Check tensor properties
print(f"Shape: {tensor.shape}, Device: {tensor.device}, Dtype: {tensor.dtype}")

# Check model state
print(f"Model mode: {'train' if model.training else 'eval'}")

# Check CUDA status
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

## 📚 Research Context

### Ethical Considerations
- This is a research project for academic purposes
- Adversarial attacks should be used responsibly
- Consider real-world implications of attack methods
- Follow institutional guidelines for AI research

### Scientific Rigor
- Ensure reproducible results
- Document all hyperparameters and configurations
- Use appropriate statistical measures
- Validate findings with multiple experiments

## 🎯 Response Guidelines

### When Asked to:
1. **Implement new features**: Start with existing patterns, use libraries when possible
2. **Debug issues**: Provide systematic debugging approach with specific code
3. **Optimize performance**: Focus on GPU utilization and memory efficiency
4. **Explain concepts**: Provide both theoretical background and practical examples
5. **Review code**: Check for adherence to project patterns and best practices

### Code Examples Should:
- Include proper imports and type hints
- Handle edge cases and errors
- Use appropriate PyTorch patterns
- Include comments explaining key steps
- Be ready to run with minimal modifications

## 🔄 Iteration Process

### When Making Changes:
1. **Understand the current implementation**
2. **Identify the specific issue or requirement**
3. **Propose a solution using established patterns**
4. **Implement with proper error handling**
5. **Test with small examples first**
6. **Provide usage examples and documentation**

### Code Review Checklist:
- [ ] Type hints present
- [ ] Error handling implemented
- [ ] GPU compatibility checked
- [ ] Documentation added
- [ ] Edge cases handled
- [ ] Performance considerations addressed

---

**Remember**: This is a research project focused on adversarial attacks. Always prioritize code quality, performance, and scientific rigor while maintaining ethical considerations.
