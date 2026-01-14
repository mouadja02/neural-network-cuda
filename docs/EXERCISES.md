# Weekly Exercises - Neural Networks From Scratch

Complete exercises organized by week, building from foundations to GPU-accelerated neural networks.

---

## Week 1: Matrix Operations (Pure Python)

**Goal**: Build a Matrix class without NumPy

**File**: `python/core/matrix.py`

### Implementation

```python
class Matrix:
    def __init__(self, data):
        """Initialize from 2D list"""
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) if data else 0
    
    def shape(self):
        return (self.rows, self.cols)
    
    def __add__(self, other):
        """Element-wise addition"""
        result = [[self.data[i][j] + other.data[i][j] 
                   for j in range(self.cols)] 
                  for i in range(self.rows)]
        return Matrix(result)
    
    def dot(self, other):
        """Matrix multiplication"""
        result = [[sum(self.data[i][k] * other.data[k][j] 
                      for k in range(self.cols))
                   for j in range(other.cols)]
                  for i in range(self.rows)]
        return Matrix(result)
    
    def T(self):
        """Transpose"""
        result = [[self.data[i][j] for i in range(self.rows)]
                  for j in range(self.cols)]
        return Matrix(result)
```

### Test Cases

```python
# Test 1: Addition
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])
C = A + B
assert C.data == [[6, 8], [10, 12]]

# Test 2: Multiplication
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])
C = A.dot(B)
assert C.data == [[19, 22], [43, 50]]

# Test 3: Transpose
A = Matrix([[1, 2, 3], [4, 5, 6]])
B = A.T()
assert B.data == [[1, 4], [2, 5], [3, 6]]
```

**Checkpoint**: All tests pass ✅

---

## Week 2: Activation & Loss Functions

**Goal**: Implement activation and loss functions with derivatives

**Files**: `python/core/activations.py`, `python/core/loss.py`

### Activations

```python
import math

def sigmoid(x):
    """Sigmoid activation: 1 / (1 + e^-x)"""
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    """Derivative: sigmoid(x) * (1 - sigmoid(x))"""
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    """ReLU: max(0, x)"""
    return max(0, x)

def relu_derivative(x):
    """Derivative: 1 if x > 0 else 0"""
    return 1.0 if x > 0 else 0.0

def softmax(x_list):
    """Softmax with numerical stability"""
    max_x = max(x_list)
    exp_x = [math.exp(x - max_x) for x in x_list]
    sum_exp = sum(exp_x)
    return [e / sum_exp for e in exp_x]
```

### Loss Functions

```python
def mean_squared_error(y_true, y_pred):
    """MSE: (1/n) * sum((y_true - y_pred)^2)"""
    n = len(y_true)
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n

def mse_derivative(y_true, y_pred):
    """Derivative: 2 * (y_pred - y_true) / n"""
    n = len(y_true)
    return [2 * (yp - yt) / n for yt, yp in zip(y_true, y_pred)]

def cross_entropy_loss(y_true, y_pred):
    """Cross-entropy: -sum(y_true * log(y_pred))"""
    epsilon = 1e-10
    return -sum(yt * math.log(yp + epsilon) 
                for yt, yp in zip(y_true, y_pred))
```

**Checkpoint**: Functions work correctly ✅

---

## Week 3: Neural Network Layer

**Goal**: Implement a dense layer with forward and backward pass

**File**: `python/core/layer.py`

### Implementation

```python
import random
import math

class DenseLayer:
    def __init__(self, input_size, output_size):
        # Xavier initialization
        limit = math.sqrt(6 / (input_size + output_size))
        self.W = [[random.uniform(-limit, limit) 
                   for _ in range(output_size)]
                  for _ in range(input_size)]
        self.b = [0.0] * output_size
        self.X = None
    
    def forward(self, X):
        """Forward pass: Y = X @ W + b"""
        self.X = X  # Save for backward pass
        
        # Matrix multiply: X @ W
        result = []
        for x_row in X:
            row = []
            for j in range(len(self.W[0])):
                val = sum(x_row[i] * self.W[i][j] 
                         for i in range(len(x_row)))
                row.append(val + self.b[j])
            result.append(row)
        return result
    
    def backward(self, dL_dout, learning_rate):
        """Backward pass: compute gradients and update weights"""
        # dL_dW = X.T @ dL_dout
        dL_dW = [[sum(self.X[k][i] * dL_dout[k][j] 
                     for k in range(len(self.X)))
                  for j in range(len(dL_dout[0]))]
                 for i in range(len(self.X[0]))]
        
        # dL_db = sum(dL_dout, axis=0)
        dL_db = [sum(dL_dout[k][j] for k in range(len(dL_dout)))
                 for j in range(len(dL_dout[0]))]
        
        # dL_dX = dL_dout @ W.T
        dL_dX = [[sum(dL_dout[i][k] * self.W[j][k] 
                     for k in range(len(dL_dout[0])))
                  for j in range(len(self.W))]
                 for i in range(len(dL_dout))]
        
        # Update weights
        for i in range(len(self.W)):
            for j in range(len(self.W[0])):
                self.W[i][j] -= learning_rate * dL_dW[i][j]
        
        for j in range(len(self.b)):
            self.b[j] -= learning_rate * dL_db[j]
        
        return dL_dX
```

**Checkpoint**: XOR problem solved ✅

---

## Week 4-5: NumPy Optimization

**Goal**: Port to NumPy and train on MNIST

**File**: `python/NeuralNetwork.py`

### Key Changes

```python
import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size):
        limit = np.sqrt(6 / (input_size + output_size))
        self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        self.b = np.zeros(output_size)
    
    def forward(self, X):
        self.X = X
        return X @ self.W + self.b
    
    def backward(self, dL_dout, learning_rate):
        dL_dW = self.X.T @ dL_dout
        dL_db = np.sum(dL_dout, axis=0)
        dL_dX = dL_dout @ self.W.T
        
        self.W -= learning_rate * dL_dW
        self.b -= learning_rate * dL_db
        
        return dL_dX
```

**Performance**: 100-1000x faster than pure Python

**Checkpoint**: MNIST >90% accuracy ✅

---

## Week 9-11: CUDA Basics

**Goal**: Implement matrix multiplication on GPU

**File**: `cuda/core/matmul.cu`

### Vector Addition (Warm-up)

```cuda
__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

### Matrix Multiplication (Tiled)

```cuda
#define TILE_SIZE 16

__global__ void matmul_tiled(
    const float *A, const float *B, float *C,
    int M, int N, int K
) {
    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < num_tiles; t++) {
        // Load tiles
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < N) {
            A_tile[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        } else {
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        int b_row = t * TILE_SIZE + threadIdx.y;
        if (b_row < N && col < K) {
            B_tile[threadIdx.y][threadIdx.x] = B[b_row * K + col];
        } else {
            B_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += A_tile[threadIdx.y][i] * B_tile[i][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}
```

**Checkpoint**: 20-50x speedup vs CPU ✅

---

## Week 12-16: Complete GPU Training

**Goal**: Full neural network training on GPU

**File**: `python/NeuralNetwork.py`

### Complete Training Loop

```python
class NeuralNetworkGPU:
    def train_batch(self, d_X, d_y, learning_rate):
        # Forward pass
        d_predictions = self.forward(d_X)
        
        # Compute loss
        loss = compute_loss(d_predictions, d_y)
        
        # Backward pass
        grad = softmax_cross_entropy_backward(d_predictions, d_y)
        grad = self.layer2.backward(grad, learning_rate)
        grad = relu_backward(grad, self.layer1_output)
        grad = self.layer1.backward(grad, learning_rate)
        
        return loss
```

**Checkpoint**: MNIST >95% accuracy in <30s ✅

---

## Summary of Achievements

| Week | Topic | Deliverable |
|------|-------|-------------|
| 1 | Matrix Operations | Pure Python Matrix class |
| 2 | Activations & Loss | All functions + derivatives |
| 3 | Neural Layer | Forward + backward pass |
| 4-5 | NumPy | MNIST >90% accuracy |
| 6-8 | C Implementation | Python C extensions |
| 9-11 | CUDA Basics | Optimized matmul |
| 12-16 | GPU Training | Complete MNIST on GPU |
| 17-18 | Visualization | Training dashboard |

---

**Total Learning Time**: ~200-300 hours  
**Final Result**: Complete understanding of neural networks from scratch!

*"The journey of a thousand miles begins with a single step."* - Lao Tzu
