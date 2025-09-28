# Transferability Attack Usage Guide

This guide shows how to use the transferability attack implementations for testing how adversarial examples transfer between different models and attacks.

## Quick Start

### 1. Basic Model-to-Model Transferability

```python
from imagenette_lab.imagenette_transferability_attacks import imagenette_transferability_model2model_in_memory
from domain.model.model_names import ModelNames
from attacks.attack_names import AttackNames

# Test how adversarial examples from one model transfer to another
results = imagenette_transferability_model2model_in_memory(
    model_names=[ModelNames().resnet18, ModelNames().densenet121],
    attack_names=[AttackNames().FGSM, AttackNames().PGD],
    images_per_attack=50,
    batch_size=1,
    results_folder="results/transferability"
)

# Print results
for result in results:
    print(f"{result.source_model} → {result.target_model} ({result.attack_name}): "
          f"{result.transfer_success}/{result.source_success} ({result.transfer_rate:.1%})")
```

### 2. Attack-to-Model Transferability

```python
from imagenette_lab.imagenette_transferability_attacks import imagenette_transferability_attack2model_in_memory

# Test how each attack transfers across different models
results = imagenette_transferability_attack2model_in_memory(
    model_names=[ModelNames().resnet18, ModelNames().densenet121, ModelNames().mobilenet_v2],
    attack_names=[AttackNames().FGSM, AttackNames().PGD, AttackNames().BIM],
    images_per_attack=50,
    batch_size=1,
    results_folder="results/transferability"
)

# Print results
for result in results:
    print(f"{result.attack_name} → {result.target_model}: "
          f"{result.transfer_success}/{result.source_success} ({result.transfer_rate:.1%})")
```

## Test Scripts

### Quick Test
```bash
python quick_test_transferability.py
```

### Comprehensive Test Suite
```bash
python test_transferability_attacks.py
```

### Interactive Examples
```bash
jupyter notebook transferability_examples.ipynb
```

## Parameters

### `model_names`
- **Type**: List[str]
- **Description**: List of model names to test
- **Example**: `[ModelNames().resnet18, ModelNames().densenet121]`
- **Available Models**: 
  - `ModelNames().resnet18`
  - `ModelNames().densenet121`
  - `ModelNames().mobilenet_v2`
  - `ModelNames().efficientnet_b0`
  - `ModelNames().vgg16`
  - And more...

### `attack_names`
- **Type**: List[str]
- **Description**: List of attack names to test
- **Example**: `[AttackNames().FGSM, AttackNames().PGD]`
- **Available Attacks**:
  - `AttackNames().FGSM`
  - `AttackNames().PGD`
  - `AttackNames().BIM`
  - `AttackNames().FFGSM`
  - `AttackNames().TPGD`
  - And more...

### `images_per_attack`
- **Type**: int
- **Description**: Number of successful adversarial examples to generate per attack
- **Default**: 100
- **Note**: Only successful attacks (that fool the source model) are counted

### `batch_size`
- **Type**: int
- **Description**: Batch size for processing images
- **Default**: 1
- **Recommendation**: Use 1-4 depending on GPU memory

### `results_folder`
- **Type**: str
- **Description**: Folder to save CSV results and failure logs
- **Default**: "results/transferability"

## Output Files

### CSV Results
- `model2model_transferability_in_memory.csv`
- `attack2model_transferability_in_memory.csv`

### Failure Logs
- `failure_logs/failure_log_[details]_[timestamp].txt`

## Understanding Results

### TransferabilityResult Object
```python
result.source_model        # Source model name
result.target_model        # Target model name
result.attack_name         # Attack name
result.source_success      # Number of successful attacks on source model
result.transfer_success    # Number of successful transfers to target model
result.total_images        # Total number of images processed
result.transfer_rate       # transfer_success / source_success
result.source_success_rate # source_success / total_images
```

### Key Metrics
- **Transfer Rate**: Percentage of successful attacks that also fool the target model
- **Source Success Rate**: Percentage of attacks that successfully fool the source model
- **Transfer Success**: Absolute number of successful transfers

## Examples

### Example 1: Minimal Test
```python
# Test with minimal setup
results = imagenette_transferability_model2model_in_memory(
    model_names=[ModelNames().resnet18, ModelNames().densenet121],
    attack_names=[AttackNames().FGSM],
    images_per_attack=10,
    batch_size=1,
    results_folder="test_results"
)
```

### Example 2: Comprehensive Analysis
```python
# Test with multiple models and attacks
model_names = [
    ModelNames().resnet18,
    ModelNames().densenet121,
    ModelNames().mobilenet_v2,
    ModelNames().efficientnet_b0
]

attack_names = [
    AttackNames().FGSM,
    AttackNames().PGD,
    AttackNames().BIM,
    AttackNames().FFGSM
]

results = imagenette_transferability_model2model_in_memory(
    model_names=model_names,
    attack_names=attack_names,
    images_per_attack=100,
    batch_size=2,
    results_folder="results/comprehensive_analysis"
)
```

### Example 3: Custom Analysis
```python
# Custom model and attack selection
custom_models = [ModelNames().resnet18, ModelNames().efficientnet_b0]
custom_attacks = [AttackNames().FGSM, AttackNames().PGD]

results = imagenette_transferability_attack2model_in_memory(
    model_names=custom_models,
    attack_names=custom_attacks,
    images_per_attack=50,
    batch_size=4,
    results_folder="results/custom_analysis"
)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` to 1
   - Reduce `images_per_attack`
   - Use fewer models/attacks

2. **No Results Generated**
   - Check if models exist in `models/imagenette/`
   - Verify attack names are correct
   - Check failure logs in `results_folder/failure_logs/`

3. **Slow Performance**
   - Increase `batch_size` if memory allows
   - Use fewer models/attacks for initial testing
   - Consider using file-based methods for large-scale analysis

### Debug Mode
```python
# Use minimal parameters for debugging
results = imagenette_transferability_model2model_in_memory(
    model_names=[ModelNames().resnet18],
    attack_names=[AttackNames().FGSM],
    images_per_attack=5,
    batch_size=1,
    results_folder="debug_results"
)
```

## Performance Tips

1. **Start Small**: Begin with 1-2 models and 1-2 attacks
2. **Batch Size**: Use batch_size=1 for debugging, increase for production
3. **Images per Attack**: Start with 10-20, increase for final analysis
4. **Memory Management**: Monitor GPU memory usage
5. **Parallel Processing**: Run different model combinations in parallel

## File-Based Transferability

For large-scale analysis using pre-saved adversarial images, use the file-based methods:

```python
from imagenette_lab.imagenette_transferability_attacks import (
    imagenette_transferability_model2model_from_files,
    imagenette_transferability_attack2model_from_files
)

# Use saved adversarial images
results = imagenette_transferability_model2model_from_files(
    model_names=model_names,
    attack_names=attack_names,
    attacked_images_folder="data/attacks/imagenette_models",
    results_folder="results/file_based"
)
```

This is useful when you have already generated and saved adversarial examples using `imagenette_direct_attacks.py`.
