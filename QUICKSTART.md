# Quick Start Guide

Get up and running with the Neural Network from Scratch project in minutes!

---

## ⚡ 5-Minute Demo

### Step 1: Setup Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/NeuralNetwork-foundations.git
cd NeuralNetwork-foundations

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt 
```

### Step 2: Run MNIST Training

```bash
cd python
python MNIST.py
```

**Expected Output**:
```
======================================================================
 MNIST Digit Classification - Full Dataset
======================================================================

Downloading MNIST Dataset...
✓ Dataset downloaded successfully!

Loading Full MNIST Dataset...
  Training set: (60000, 784)
  Test set: (10000, 784)

Training Started...
Epoch  1/10 | Loss: 0.4521 | Train Acc: 87.32% | Test Acc: 88.45% | Time: 2.8s
Epoch  2/10 | Loss: 0.2134 | Train Acc: 93.21% | Test Acc: 93.67% | Time: 2.7s
...
Epoch 10/10 | Loss: 0.0823 | Train Acc: 97.45% | Test Acc: 95.23% | Time: 2.6s

Training Complete! 🎉
Final Test Accuracy: 95.23%
```

**Time**: ~30 seconds on RTX 3080

---

## 📖 Understanding the Code

### 1. **Pure Python Implementation**

```python
# python/core/matrix.py
class Matrix:
    def dot(self, other):
        """Matrix multiplication - O(n³) algorithm"""
        result = [[sum(self.data[i][k] * other.data[k][j] 
                      for k in range(self.cols))
                   for j in range(other.cols)]
                  for i in range(self.rows)]
        return Matrix(result)
```

### 2. **CUDA Kernel**

```cuda
// python/cuda/matmul.cu
__global__ void matmul_tiled(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float A_tile[16][16];
    __shared__ float B_tile[16][16];
    
    // Tiled matrix multiplication
    // 10-20x faster than naive implementation
}
```

### 3. **Python API**

```python
# python/NeuralNetwork.py
from NeuralNetwork import NeuralNetworkGPU

nn = NeuralNetworkGPU(784, 128, 10)
loss = nn.train_batch(X_gpu, y_gpu, learning_rate=0.01)
accuracy = nn.evaluate(X_test, y_test)
```

---

## 🎯 What to Show

### For Technical Interview

1. **Architecture Overview**
   - Show `README.md` - project structure
   - Explain progression: Python → NumPy → C → CUDA

2. **Code Deep Dive**
   - `python/core/layer.py` - backpropagation implementation
   - `cuda/core/matmul.cu` - tiled matrix multiplication
   - `python/NeuralNetwork.py` - complete GPU neural network

3. **Performance Benchmarks**
   - Matrix multiplication: 56,000x speedup
   - MNIST training: 2 hours → 30 seconds

4. **Technical Challenges**
   - Memory coalescing in CUDA
   - Shared memory optimization
   - Gradient computation on GPU

### For Portfolio

1. **Show Live Demo**
   ```bash
   python MNIST.py
   ```

2. **Explain Learning Journey**
   - Started with pure Python (no libraries)
   - Optimized with NumPy (100-1000x faster)
   - Implemented in C (understanding memory)
   - Accelerated with CUDA (20-50x faster)

3. **Highlight Achievements**
   - ✅ 95%+ accuracy on MNIST
   - ✅ <30s training time
   - ✅ Complete understanding (no black boxes)

---

## 📚 Documentation Structure

```
docs/
├── LEARNING_PATH.md    # 18-week curriculum
├── CUDA_GUIDE.md       # GPU programming guide
├── EXERCISES.md        # Week-by-week exercises
└── archive/            # Old documentation
```

**Start here**: `docs/LEARNING_PATH.md`

---

## 🔧 Troubleshooting

### CUDA Not Found

```bash
# Check CUDA installation
nvcc --version

# If not installed:
# Download from: https://developer.nvidia.com/cuda-downloads
```

### PyCUDA Installation Issues

```bash
# Windows
pip install pycuda

# Linux
sudo apt-get install nvidia-cuda-toolkit
pip install pycuda
```

### Import Errors

```bash
# Make sure you're in the python/ directory
cd python

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/NeuralNetwork-foundations/python"
```

---

## 🎓 Learning Path

### Beginner (Week 1-3)
- Start with `docs/EXERCISES.md` - Week 1
- Implement Matrix class in pure Python
- Understand forward/backward propagation

### Intermediate (Week 4-8)
- Port to NumPy
- Train on MNIST
- Implement in C

### Advanced (Week 9-16)
- Learn CUDA programming
- Optimize matrix multiplication
- Complete GPU training

---

## 🚀 Next Steps

After running the demo:

1. **Explore the Code**
   - Read `python/NeuralNetwork.py`
   - Study `cuda/core/matmul.cu`
   - Understand backpropagation

2. **Read Documentation**
   - `docs/LEARNING_PATH.md` - full curriculum
   - `docs/CUDA_GUIDE.md` - GPU programming

3. **Experiment**
   - Modify network architecture
   - Try different learning rates
   - Add more layers
