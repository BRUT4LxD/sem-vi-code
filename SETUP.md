# 🚀 Project Setup Guide

This guide will help you set up the adversarial attack research environment from scratch.

## 📋 Prerequisites

### System Requirements
- **OS**: Windows 10/11, Linux, or macOS
- **Python**: 3.8+ (recommended: 3.9 or 3.10)
- **CUDA**: 11.8+ (for GPU acceleration)
- **RAM**: 8GB+ (16GB+ recommended)
- **Storage**: 10GB+ free space

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (RTX series recommended)
- **VRAM**: 4GB+ (8GB+ recommended for large models)

## 🔧 Installation Steps

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd sem-vi-code
```

### 2. Check CUDA Installation
Verify CUDA is properly installed:
```bash
# Check NVIDIA driver and CUDA version
nvidia-smi

# Check CUDA compiler version
nvcc --version
```

**Expected Output:**
- NVIDIA driver version
- CUDA version (11.8+)
- GPU information (RTX series recommended)

### 3. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 4. Upgrade pip and Install Dependencies
```bash
# Upgrade pip to latest version
python -m pip install --upgrade pip

# Install PyTorch with CUDA support
# For CUDA 11.8:
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### 5. Verify Installation
```bash
# Test CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Test scikit-learn
python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"
```

**Expected Output:**
```
CUDA available: True
CUDA version: 11.8
GPU count: 1
scikit-learn version: 1.3.0
```

## 🧪 Testing the Setup

### 1. Run Basic Test
```bash
python -c "
import torch
import numpy as np
from sklearn.metrics import accuracy_score

# Test PyTorch
x = torch.randn(10, 3).cuda()
print('✅ PyTorch CUDA test passed')

# Test scikit-learn
y_true = np.array([0, 1, 2, 0, 1])
y_pred = np.array([0, 1, 2, 0, 1])
acc = accuracy_score(y_true, y_pred)
print(f'✅ scikit-learn test passed: accuracy = {acc}')
"
```

### 2. Test Project Modules
```bash
python -c "
from evaluation.metrics import Metrics
from domain.attack.attack_result import AttackResult
print('✅ Project modules imported successfully')
"
```

## 📁 Project Structure

```
sem-vi-code/
├── architectures/          # Neural network architectures
├── attacks/               # Adversarial attack implementations
│   ├── white_box/        # White-box attacks
│   └── black_box/        # Black-box attacks
├── config/               # Configuration files
├── data/                 # Datasets (excluded from git)
├── data_eng/            # Data engineering utilities
├── domain/              # Domain models and entities
├── evaluation/          # Evaluation metrics and validation
├── models/              # Trained models (excluded from git)
├── results/             # Experiment results (excluded from git)
├── runs/                # Training runs (excluded from git)
├── shared/              # Shared utilities
├── training/            # Training scripts
├── requirements.txt     # Python dependencies
├── SETUP.md            # This file
└── README.md           # Project documentation
```

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Not Available
**Problem**: `torch.cuda.is_available()` returns `False`

**Solutions**:
- Verify NVIDIA driver is installed: `nvidia-smi`
- Check CUDA version compatibility
- Reinstall PyTorch with correct CUDA version
- Restart your system after driver installation

#### 2. Import Errors
**Problem**: Module import failures

**Solutions**:
- Ensure virtual environment is activated
- Check Python path: `python -c "import sys; print(sys.path)"`
- Reinstall dependencies: `pip install -r requirements.txt`

#### 3. Memory Issues
**Problem**: CUDA out of memory errors

**Solutions**:
- Reduce batch size in training scripts
- Use gradient accumulation
- Clear GPU cache: `torch.cuda.empty_cache()`

#### 4. Version Conflicts
**Problem**: Package version conflicts

**Solutions**:
- Create fresh virtual environment
- Use specific package versions in requirements.txt
- Check compatibility matrix

### Getting Help

1. **Check logs**: Look for error messages in terminal output
2. **Verify versions**: Ensure all packages are compatible
3. **Test incrementally**: Test each component separately
4. **Check documentation**: Refer to PyTorch and scikit-learn docs

## 🚀 Quick Start

Once setup is complete, you can start using the project:

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# Run a simple attack
python attacks_lab.ipynb

# Train a model
python training/train.py

# Evaluate attacks
python evaluation/validation.py
```

## 📚 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
- [Adversarial Attacks Research](https://adversarial-attacks-pytorch.readthedocs.io/)

---

**Note**: This project is designed for research purposes. Ensure you have proper permissions and follow ethical guidelines when conducting adversarial attack experiments.
