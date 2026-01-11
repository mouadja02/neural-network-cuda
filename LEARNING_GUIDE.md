# Neural Network From Scratch - Learning Journey
**Goal**: Build a C+CUDA neural network library for digit recognition (0-9), understanding every detail from linear algebra to GPU computing.

---

## 📚 Learning Philosophy
- **No Black Boxes**: Implement everything yourself
- **Test Everything**: Each component must be verifiable
- **Iterate**: Start simple, add complexity gradually
- **Debug Visually**: Build tools to see what's happening

---

## 🎯 Phase 1: Foundation - Pure Python (Weeks 1-3)

### Exercise 1.1: Matrix Operations from Scratch
**File**: `python/core/matrix.py`

Build a Matrix class WITHOUT numpy. Implement:
```python
class Matrix:
    def __init__(self, data)  # 2D list
    def shape(self)           # returns (rows, cols)
    def __add__(self, other)  # matrix addition
    def __sub__(self, other)  # matrix subtraction
    def __mul__(self, other)  # element-wise multiplication
    def dot(self, other)      # matrix multiplication
    def T(self)               # transpose
    def apply(self, func)     # apply function to each element
```

**Test Cases** (create `python/tests/test_matrix.py`):
- Add two 2x2 matrices
- Multiply 3x2 with 2x4 matrix
- Transpose a 4x3 matrix
- Verify A @ B != B @ A (matrix multiplication is not commutative)

**Resources**:
- Khan Academy: Matrix Multiplication
- 3Blue1Brown: "Essence of Linear Algebra" (YouTube, chapters 3-4)

**Success Criteria**: All tests pass, understand why matrix multiplication works this way

---

### Exercise 1.2: Activation Functions
**File**: `python/core/activations.py`

Implement these functions and their derivatives:
```python
def sigmoid(x):
    # f(x) = 1 / (1 + e^-x)
    pass

def sigmoid_derivative(x):
    # f'(x) = f(x) * (1 - f(x))
    pass

def relu(x):
    # f(x) = max(0, x)
    pass

def relu_derivative(x):
    # f'(x) = 1 if x > 0 else 0
    pass

def softmax(x):
    # f(x_i) = e^x_i / sum(e^x_j)
    # Numerical stability: subtract max before exp
    pass
```

**Test Cases**:
- Plot sigmoid for x in [-10, 10]
- Verify sigmoid(0) = 0.5
- Check softmax output sums to 1
- Verify derivatives numerically (finite differences)

**Challenge**: Implement tanh and leaky_relu

**Resources**:
- Understand derivatives: https://www.khanacademy.org/math/calculus-1
- Visualize activation functions: https://playground.tensorflow.org

---

### Exercise 1.3: Loss Functions
**File**: `python/core/loss.py`

```python
def mean_squared_error(y_true, y_pred):
    # MSE = (1/n) * sum((y_true - y_pred)^2)
    pass

def mse_derivative(y_true, y_pred):
    pass

def cross_entropy_loss(y_true, y_pred):
    # For one-hot encoded labels
    # -sum(y_true * log(y_pred))
    # Add epsilon for numerical stability
    pass

def cross_entropy_derivative(y_true, y_pred):
    pass
```

**Test Cases**:
- MSE of identical arrays should be 0
- MSE is always non-negative
- Cross-entropy with one-hot [0,1,0] and prediction [0.1, 0.8, 0.1]

---

### Exercise 1.4: Progress Bar from Scratch
**File**: `python/utils/progress.py`

Build a simple tqdm replacement:
```python
class ProgressBar:
    def __init__(self, total, desc=""):
        # total: number of iterations
        # desc: description text
        pass

    def update(self, n=1):
        # Update progress by n steps
        # Print: [=====>    ] 50% | 50/100 | ETA: 5s
        pass

    def close(self):
        # Finish the progress bar
        pass
```

**Features to implement**:
- Percentage complete
- Visual bar (20 characters wide)
- Estimated time remaining
- Smooth updates (don't print every iteration)

**Test**: Use it in a loop with `time.sleep(0.1)`

---

### Exercise 1.5: Your First Neural Network Layer
**File**: `python/core/layer.py`

```python
class DenseLayer:
    def __init__(self, input_size, output_size):
        # Initialize weights: Xavier initialization
        # W shape: (input_size, output_size)
        # b shape: (output_size,)
        pass

    def forward(self, X):
        # X @ W + b
        # Save X for backward pass
        pass

    def backward(self, dL_dout, learning_rate):
        # Compute gradients:
        # dL_dW = X.T @ dL_dout
        # dL_db = sum(dL_dout, axis=0)
        # dL_dX = dL_dout @ W.T
        # Update weights: W -= learning_rate * dL_dW
        # Return dL_dX for previous layer
        pass
```

**Test**: Create a 3->5->2 network and verify shapes at each step

**Resources**:
- Backpropagation calculus: 3Blue1Brown "Neural Networks" Chapter 3-4
- Matrix calculus: https://explained.ai/matrix-calculus/

---

### 🎓 Checkpoint 1 (End of Week 3)
Build a complete 2-layer network (Python only) that learns XOR:

**File**: `python/examples/xor_example.py`

```python
# XOR dataset
X = [[0,0], [0,1], [1,0], [1,1]]
y = [[0], [1], [1], [0]]

# Network: 2 -> 4 -> 1
# Train for 10,000 epochs
# Should achieve >95% accuracy
```

---

## 🔥 Phase 2: Optimize with NumPy (Weeks 4-5)

### Exercise 2.1: Port to NumPy
**File**: `python/core/matrix_numpy.py`

Reimplement your Matrix class using NumPy arrays:
- Benchmark: Compare speed of pure Python vs NumPy for 1000x1000 matrix multiply
- Profile: Use `cProfile` to find bottlenecks

**Expected speedup**: 100-1000x faster

---

### Exercise 2.2: Mini-Batch Training
**File**: `python/core/network.py`

```python
class NeuralNetwork:
    def train_batch(self, X_batch, y_batch, learning_rate):
        # X_batch: (batch_size, input_features)
        # Update with average gradient over batch
        pass
```

**Implement**:
- Stochastic Gradient Descent (batch_size=1)
- Mini-batch (batch_size=32)
- Batch (batch_size=all data)

**Compare**: Training speed and convergence for each

---

### Exercise 2.3: Data Loading Pipeline
**File**: `python/utils/data_loader.py`

```python
def load_mnist():
    # Download MNIST dataset (or use torchvision)
    # Return: X_train, y_train, X_test, y_test
    # Normalize: X /= 255.0
    # One-hot encode: y
    pass

class DataGenerator:
    def __init__(self, X, y, batch_size, shuffle=True):
        pass

    def __iter__(self):
        # Yield batches
        pass
```

---

### 🎓 Checkpoint 2 (End of Week 5)
Train a 3-layer network on MNIST:

**Target**:
- Architecture: 784 -> 128 -> 64 -> 10
- Accuracy: >90% on test set
- Training time: <5 minutes on CPU

**File**: `python/examples/mnist_numpy.py`

---

## ⚡ Phase 3: Introduction to C (Weeks 6-8)

### Exercise 3.1: Matrix in C
**File**: `c/matrix.h` and `c/matrix.c`

```c
typedef struct {
    float *data;
    int rows;
    int cols;
} Matrix;

Matrix* matrix_create(int rows, int cols);
void matrix_free(Matrix *m);
Matrix* matrix_multiply(Matrix *a, Matrix *b);
Matrix* matrix_add(Matrix *a, Matrix *b);
Matrix* matrix_transpose(Matrix *m);
void matrix_print(Matrix *m);
```

**Test**: `c/tests/test_matrix.c` using assertions

**Build System**: Create `Makefile`
```makefile
CC = gcc
CFLAGS = -Wall -O2 -lm

test_matrix: tests/test_matrix.c matrix.c
	$(CC) $(CFLAGS) -o test_matrix tests/test_matrix.c matrix.c
```

---

### Exercise 3.2: Python C Extension
**File**: `c/python_bindings/matrix_module.c`

Create a Python module that uses your C matrix code:

```c
#include <Python.h>
#include "../matrix.h"

static PyObject* py_matrix_multiply(PyObject* self, PyObject* args) {
    // Convert Python lists to Matrix
    // Call matrix_multiply
    // Convert back to Python list
}

static PyMethodDef MatrixMethods[] = {
    {"multiply", py_matrix_multiply, METH_VARARGS, "Multiply matrices"},
    {NULL, NULL, 0, NULL}
};
```

**File**: `setup.py`
```python
from distutils.core import setup, Extension

module = Extension('cmatrix', sources=['c/python_bindings/matrix_module.c', 'c/matrix.c'])
setup(name='cmatrix', ext_modules=[module])
```

**Test in Python**:
```python
import cmatrix
result = cmatrix.multiply([[1,2],[3,4]], [[5,6],[7,8]])
```

**Resources**:
- Python C API: https://docs.python.org/3/extending/extending.html

---

### Exercise 3.3: Memory Management & Debugging
**File**: `c/utils/memory_pool.c`

Implement a simple memory pool for faster allocation:
```c
typedef struct {
    float *pool;
    size_t size;
    size_t used;
} MemoryPool;

MemoryPool* pool_create(size_t size);
float* pool_alloc(MemoryPool *pool, size_t n);
void pool_reset(MemoryPool *pool);
void pool_free(MemoryPool *pool);
```

**Debug with**:
- Valgrind (Linux/WSL): `valgrind --leak-check=full ./test_matrix`
- AddressSanitizer: `gcc -fsanitize=address`

---

### 🎓 Checkpoint 3 (End of Week 8)
Implement a full feedforward pass in C:

**File**: `c/examples/forward_pass.c`
- 784 -> 128 -> 10 network
- Load weights from Python-trained model
- Run inference on 1000 MNIST images
- Compare outputs with Python version (should match within 1e-5)

---

## 🚀 Phase 4: CUDA Fundamentals (Weeks 9-11)

### Exercise 4.1: Hello CUDA
**File**: `cuda/hello.cu`

```cuda
#include <stdio.h>

__global__ void hello_kernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from thread %d\n", idx);
}

int main() {
    hello_kernel<<<2, 4>>>();  // 2 blocks, 4 threads each
    cudaDeviceSynchronize();
    return 0;
}
```

**Compile**: `nvcc -o hello hello.cu`

**Resources**:
- CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Udacity: Intro to Parallel Programming (free course)

---

### Exercise 4.2: Vector Addition on GPU
**File**: `cuda/vector_add.cu`

```cuda
__global__ void vector_add_kernel(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void vector_add_gpu(float *h_a, float *h_b, float *h_c, int n) {
    float *d_a, *d_b, *d_c;

    // 1. Allocate device memory
    cudaMalloc(&d_a, n * sizeof(float));
    // ... allocate d_b, d_c

    // 2. Copy host to device
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    // ... copy b

    // 3. Launch kernel
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    vector_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);

    // 4. Copy device to host
    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    // 5. Free device memory
    cudaFree(d_a);
    // ... free others
}
```

**Test**: Compare with CPU version for 1M elements

---

### Exercise 4.3: Matrix Multiplication - Naive
**File**: `cuda/matmul_naive.cu`

```cuda
__global__ void matmul_naive(float *A, float *B, float *C, int M, int N, int K) {
    // C[M x K] = A[M x N] * B[N x K]
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}
```

**Benchmark**: 1024x1024 matrices
- CPU (single-threaded)
- CPU (NumPy)
- GPU (naive)

---

### Exercise 4.4: Matrix Multiplication - Optimized
**File**: `cuda/matmul_shared.cu`

Use shared memory and tiling:

```cuda
#define TILE_SIZE 16

__global__ void matmul_tiled(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < N)
            A_tile[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < K && t * TILE_SIZE + threadIdx.y < N)
            B_tile[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * K + col];
        else
            B_tile[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial sum
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

**Expected speedup**: 10-20x over naive version

**Resources**:
- NVIDIA: "Better Performance at Lower Occupancy"
- Shared memory optimization guide

---

### Exercise 4.5: CUDA Profiling
Use `nvprof` or Nsight Compute to analyze:
- Kernel execution time
- Memory bandwidth utilization
- Occupancy
- Bank conflicts

**Goal**: Understand what limits your kernel performance

---

### 🎓 Checkpoint 4 (End of Week 11)
Implement forward pass on GPU:

**File**: `cuda/examples/forward_pass.cu`
- Matrix multiply for each layer
- Activation functions (sigmoid, ReLU)
- Benchmark vs CPU version

**Target speedup**: 20-50x on RTX 3080

---

## 🧠 Phase 5: Complete Neural Network (Weeks 12-16)

### Exercise 5.1: Backward Pass on GPU
**File**: `cuda/backward.cu`

Implement gradient computation:
```cuda
__global__ void backward_dense_layer(
    float *dL_dout,     // Gradient from next layer
    float *X,           // Input to this layer
    float *W,           // Weights
    float *dL_dW,       // Output: gradient of weights
    float *dL_db,       // Output: gradient of biases
    float *dL_dX,       // Output: gradient of inputs
    int batch_size, int input_size, int output_size
);
```

**Challenges**:
- Reduction for bias gradients (use `atomicAdd`)
- Large matrix transposes
- Memory access patterns

---

### Exercise 5.2: Activation Function Kernels
**File**: `cuda/activations.cu`

```cuda
__global__ void relu_forward(float *input, float *output, int n);
__global__ void relu_backward(float *grad_output, float *input, float *grad_input, int n);
__global__ void sigmoid_forward(float *input, float *output, int n);
__global__ void sigmoid_backward(float *grad_output, float *output, float *grad_input, int n);
__global__ void softmax_forward(float *input, float *output, int batch_size, int n);
```

**Softmax challenge**: Requires reduction (max, sum) within each row

---

### Exercise 5.3: SGD Optimizer
**File**: `cuda/optimizer.cu`

```cuda
__global__ void sgd_update(float *weights, float *gradients, float learning_rate, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        weights[idx] -= learning_rate * gradients[idx];
    }
}
```

**Bonus**: Implement momentum, Adam optimizer

---

### Exercise 5.4: Complete Training Loop
**File**: `cuda/train.cu`

Full pipeline:
1. Load data on GPU
2. Forward pass
3. Compute loss
4. Backward pass
5. Update weights
6. Track metrics

**File**: `cuda/utils/metrics.cu`
- Accuracy calculation
- Loss tracking
- Confusion matrix

---

### Exercise 5.5: Python Wrapper
**File**: `python/cuda_nn/__init__.py`

Create a clean Python API:
```python
import cuda_nn

model = cuda_nn.Sequential([
    cuda_nn.Dense(784, 128, activation='relu'),
    cuda_nn.Dense(128, 64, activation='relu'),
    cuda_nn.Dense(64, 10, activation='softmax')
])

model.compile(optimizer='sgd', loss='cross_entropy', learning_rate=0.01)
model.fit(X_train, y_train, epochs=10, batch_size=32)
accuracy = model.evaluate(X_test, y_test)
```

**Use ctypes or pybind11**:
```python
# Using ctypes
import ctypes
cuda_lib = ctypes.CDLL('./libcudann.so')
```

Or better, **pybind11**:
```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

PYBIND11_MODULE(cuda_nn, m) {
    m.def("train", &train_wrapper);
    m.def("predict", &predict_wrapper);
}
```

---

### 🎓 Checkpoint 5 (End of Week 16)
**FINAL PROJECT**: Complete MNIST digit classifier

**Requirements**:
- C+CUDA backend
- Python interface
- Accuracy >97% on test set
- Training time <30 seconds for 10 epochs
- Inference: <1ms per image on GPU

**Deliverables**:
1. Library: `libcudann.so` (Linux) or `cudann.dll` (Windows)
2. Python package: `pip install .`
3. Demo script: `examples/mnist_demo.py`
4. Benchmark report comparing CPU vs GPU

---

## 🎨 Phase 6: Visualization & Debugging (Weeks 17-18)

### Exercise 6.1: Network Visualizer
**File**: `python/viz/network_viz.py`

Create ASCII art of your network:
```
Input Layer (784)    Hidden Layer (128)    Output Layer (10)
    [●●●]                 [●●●]                 [●]
    [●●●]      ══════>    [●●●]      ══════>   [●]
    [●●●]                 [●●●]                 [●]

Weights: min=-0.5, max=0.8, mean=0.02
```

**Bonus**: Use matplotlib to visualize:
- Weight distributions (histograms)
- Activation maps
- Gradient flow
- Learning curves

---

### Exercise 6.2: Live Training Dashboard
**File**: `python/viz/dashboard.py`

Real-time terminal UI showing:
```
Epoch 5/10 [=====>    ] 50%
Loss: 0.234 | Accuracy: 91.5%
Learning Rate: 0.001

Layer 1 weights: [▁▂▃▅▆▇██▇▆▅▃▂▁] (histogram)
Layer 2 gradients: [▁▁▂▃▃▃▂▁▁]

GPU: 75% | Memory: 2.1GB/10GB
ETA: 2m 34s
```

**Use libraries**: `curses` (built-in) or `rich`

---

### Exercise 6.3: Gradient Checker
**File**: `python/utils/gradient_check.py`

Numerical gradient checking:
```python
def check_gradients(layer, X, epsilon=1e-5):
    """
    Compare analytical gradients with numerical approximation:
    f'(x) ≈ (f(x+ε) - f(x-ε)) / (2ε)
    """
    analytical_grad = layer.backward(X)

    numerical_grad = []
    for i in range(len(layer.weights)):
        layer.weights[i] += epsilon
        loss_plus = layer.forward(X)
        layer.weights[i] -= 2 * epsilon
        loss_minus = layer.forward(X)
        layer.weights[i] += epsilon  # restore

        numerical_grad.append((loss_plus - loss_minus) / (2 * epsilon))

    difference = norm(analytical_grad - numerical_grad)
    print(f"Gradient difference: {difference}")
    assert difference < 1e-7, "Gradients don't match!"
```

---

### Exercise 6.4: Model Introspection
**File**: `cuda/utils/debug.cu`

Debug utilities:
```cuda
// Print device array
void print_device_array(float *d_array, int n, const char *name);

// Check for NaN/Inf
__global__ void check_validity(float *array, int n, int *has_nan);

// Dump layer activations to file
void dump_activations(float *d_activations, int n, const char *filename);
```

**Use these during training to debug**:
- Exploding/vanishing gradients
- Dead neurons
- Numerical instability

---

## 🏆 Advanced Challenges (Optional)

### Challenge 1: Convolutional Layer
Implement Conv2D in CUDA:
- im2col transformation
- Or direct convolution kernel
- Backward pass for convolutions

### Challenge 2: Batch Normalization
- Running mean/variance tracking
- Training vs inference mode
- Gradient computation

### Challenge 3: Custom Optimizers
- Momentum
- RMSprop
- Adam
- Learning rate scheduling

### Challenge 4: Multi-GPU Training
- Data parallelism
- Gradient synchronization
- NCCL library usage

### Challenge 5: Inference Optimization
- Kernel fusion (combine matmul + activation)
- INT8 quantization
- TensorRT integration comparison

---

## 📊 Project Structure

```
NeuralNetwork-foundations/
├── python/
│   ├── core/
│   │   ├── matrix.py              # Ex 1.1
│   │   ├── activations.py         # Ex 1.2
│   │   ├── loss.py                # Ex 1.3
│   │   ├── layer.py               # Ex 1.5
│   │   └── network.py             # Ex 2.2
│   ├── utils/
│   │   ├── progress.py            # Ex 1.4
│   │   ├── data_loader.py         # Ex 2.3
│   │   └── gradient_check.py      # Ex 6.3
│   ├── viz/
│   │   ├── network_viz.py         # Ex 6.1
│   │   └── dashboard.py           # Ex 6.2
│   ├── tests/
│   │   └── test_*.py
│   └── examples/
│       ├── xor_example.py         # Checkpoint 1
│       └── mnist_numpy.py         # Checkpoint 2
├── c/
│   ├── matrix.h/.c                # Ex 3.1
│   ├── python_bindings/           # Ex 3.2
│   ├── utils/
│   │   └── memory_pool.c          # Ex 3.3
│   ├── tests/
│   └── examples/
│       └── forward_pass.c         # Checkpoint 3
├── cuda/
│   ├── hello.cu                   # Ex 4.1
│   ├── vector_add.cu              # Ex 4.2
│   ├── matmul_naive.cu            # Ex 4.3
│   ├── matmul_shared.cu           # Ex 4.4
│   ├── activations.cu             # Ex 5.2
│   ├── backward.cu                # Ex 5.1
│   ├── optimizer.cu               # Ex 5.3
│   ├── train.cu                   # Ex 5.4
│   ├── utils/
│   │   ├── metrics.cu
│   │   └── debug.cu               # Ex 6.4
│   └── examples/
│       └── forward_pass.cu        # Checkpoint 4
├── setup.py                       # Python package
├── Makefile                       # Build system
├── CMakeLists.txt                 # Alternative build
└── README.md
```

---

## 📚 Essential Resources

### Books
1. **"Deep Learning" by Goodfellow, Bengio, Courville** - Free online
2. **"Programming Massively Parallel Processors" by Kirk & Hwu** - CUDA bible
3. **"The Matrix Cookbook"** - Quick reference for matrix calculus

### Online Courses
1. **3Blue1Brown - Neural Networks** (YouTube)
2. **Andrej Karpathy - Neural Networks: Zero to Hero** (YouTube)
3. **Stanford CS231n** - Convolutional Neural Networks
4. **NVIDIA DLI - Fundamentals of Accelerated Computing with CUDA**

### Documentation
1. CUDA C Programming Guide
2. cuBLAS documentation
3. NumPy documentation
4. Python C API reference

### Papers
1. **"ImageNet Classification with Deep CNNs"** (AlexNet) - Understand the revolution
2. **"Adam: A Method for Stochastic Optimization"** - Modern optimizer
3. **"Batch Normalization"** - Training acceleration

---

## ✅ Weekly Milestones

| Week | Milestone | Verification |
|------|-----------|-------------|
| 1-3  | Pure Python NN | XOR solved |
| 4-5  | NumPy NN | MNIST >90% |
| 6-8  | C library + bindings | Inference matches Python |
| 9-11 | CUDA basics | Matmul 20x faster |
| 12-16| Full CUDA training | MNIST >97%, <30s |
| 17-18| Visualization | Live dashboard works |

---

## 🎯 Your First Task (Week 1)

**START HERE**:

1. Create `python/core/matrix.py` - Implement Matrix class (Ex 1.1)
2. Create `python/tests/test_matrix.py` - Write 10 test cases
3. Run tests and make them pass
4. Send me your code for review

**Questions to think about**:
- Why is matrix multiplication O(n³)?
- What's the difference between element-wise and matrix multiplication?
- How does transpose affect multiplication order?

**When you're done**: Show me your code and I'll give you feedback + next exercises!

---

## 💡 Learning Tips

1. **Code First, Optimize Later**: Make it work, then make it fast
2. **Test Everything**: If it's not tested, it's broken
3. **Visualize**: Print shapes, plot distributions, inspect values
4. **Compare**: Always verify against reference implementations
5. **Ask Questions**: No question is too basic
6. **Take Breaks**: This is a marathon, not a sprint
7. **Keep a Journal**: Document what you learn each day

---

## 🐛 Common Pitfalls

- **Shape Mismatches**: Always print matrix shapes
- **Memory Leaks**: Use Valgrind early and often
- **Numerical Instability**: Add epsilon, use log-sum-exp trick
- **Gradient Bugs**: Use gradient checking religiously
- **CUDA Race Conditions**: Understand `__syncthreads()`
- **Premature Optimization**: Profile before optimizing

---

## 📞 Getting Help

When you're stuck:
1. Show me your code
2. Describe what you expected vs what happened
3. Share error messages
4. Tell me what you've tried

I'm here to teach you, not judge you. Every expert was once a beginner!

---

**Ready to start? Let's build your first Matrix class! 🚀**
