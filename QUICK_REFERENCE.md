# Quick Reference Cheat Sheet

Fast lookup for common concepts you'll need throughout your journey.

---

## 🧮 Matrix Operations

### Matrix Shapes
```
Matrix A: (m × n) means m rows, n columns

Example: [[1, 2, 3],   <- This is (2 × 3)
          [4, 5, 6]]
```

### Addition/Subtraction (Element-wise)
```
Requirement: Same shape (m × n) + (m × n) = (m × n)

[[1, 2]  +  [[5, 6]  =  [[6, 8]
 [3, 4]]     [7, 8]]     [10, 12]]

Formula: C[i][j] = A[i][j] + B[i][j]
```

### Element-wise Multiplication (Hadamard Product)
```
Requirement: Same shape (m × n) * (m × n) = (m × n)

[[1, 2]  *  [[5, 6]  =  [[5, 12]
 [3, 4]]     [7, 8]]     [21, 32]]

Formula: C[i][j] = A[i][j] * B[i][j]
```

### Matrix Multiplication (Dot Product)
```
Requirement: (m × n) @ (n × p) = (m × p)
            Inner dimensions must match! ^^^

[[1, 2]  @  [[5, 6]  =  [[19, 22]
 [3, 4]]     [7, 8]]     [43, 50]]

Formula: C[i][j] = Σ(A[i][k] * B[k][j]) for k=0 to n-1

Code:
for i in range(m):
    for j in range(p):
        sum = 0
        for k in range(n):
            sum += A[i][k] * B[k][j]
        C[i][j] = sum
```

### Transpose
```
(m × n) → (n × m)
Flip rows and columns

[[1, 2, 3]  ->  [[1, 4]
 [4, 5, 6]]      [2, 5]
                 [3, 6]]

Formula: B[j][i] = A[i][j]
```

### Important Properties
```
(A + B)ᵀ = Aᵀ + Bᵀ           (Transpose distributes over addition)
(A * B)ᵀ = Bᵀ * Aᵀ           (Transpose reverses order!)
A @ I = A                    (Identity property)
A @ B ≠ B @ A                (NOT commutative!)
```

---

## 🎛️ Activation Functions

### Sigmoid
```python
f(x) = 1 / (1 + e^(-x))
f'(x) = f(x) * (1 - f(x))

Range: (0, 1)
Use: Binary classification, gates in LSTM
Problem: Vanishing gradients
```

### ReLU (Rectified Linear Unit)
```python
f(x) = max(0, x)
f'(x) = 1 if x > 0 else 0

Range: [0, ∞)
Use: Most common in hidden layers
Advantage: No vanishing gradient, fast
Problem: Dead neurons (always 0)
```

### Leaky ReLU
```python
f(x) = x if x > 0 else 0.01 * x
f'(x) = 1 if x > 0 else 0.01

Range: (-∞, ∞)
Use: Alternative to ReLU
Advantage: No dead neurons
```

### Tanh
```python
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
f'(x) = 1 - f(x)²

Range: (-1, 1)
Use: RNNs, zero-centered output
Problem: Vanishing gradients
```

### Softmax
```python
f(x_i) = e^(x_i) / Σ(e^(x_j))

# Numerical stability trick:
f(x_i) = e^(x_i - max(x)) / Σ(e^(x_j - max(x)))

Range: (0, 1), sum = 1
Use: Multi-class classification (output layer)
Output: Probability distribution
```

---

## 📊 Loss Functions

### Mean Squared Error (MSE)
```python
L = (1/n) * Σ(y_true - y_pred)²
dL/dy_pred = -2/n * (y_true - y_pred)

Use: Regression tasks
```

### Binary Cross-Entropy
```python
L = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
dL/dŷ = (ŷ - y) / (ŷ * (1 - ŷ))

Use: Binary classification (with sigmoid)
```

### Categorical Cross-Entropy
```python
L = -Σ(y_true * log(y_pred))

# For one-hot encoded labels
# Add epsilon for numerical stability:
L = -Σ(y_true * log(y_pred + ε))

dL/dy_pred = -y_true / y_pred

Use: Multi-class classification (with softmax)
```

---

## 🔄 Backpropagation

### Chain Rule
```
If y = f(g(x)), then dy/dx = df/dg * dg/dx

Example:
y = σ(Wx + b)
dy/dW = dy/dσ * dσ/d(Wx+b) * d(Wx+b)/dW
      = dL * σ'(z) * x
```

### Layer Gradients
```
Dense Layer: y = Wx + b

Forward:
z = Wx + b
a = activation(z)

Backward (given dL/da):
dL/dz = dL/da * activation'(z)     [element-wise]
dL/dW = xᵀ @ dL/dz                 [matrix multiply]
dL/db = sum(dL/dz, axis=0)         [sum over batch]
dL/dx = dL/dz @ Wᵀ                 [for prev layer]

Update:
W = W - learning_rate * dL/dW
b = b - learning_rate * dL/db
```

### Gradient Checking
```python
# Numerical gradient approximation
epsilon = 1e-5
numerical_grad = (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)

# Compare with analytical gradient
difference = abs(analytical_grad - numerical_grad)
assert difference < 1e-7, "Gradients don't match!"
```

---

## 🎯 Training Process

### Full Training Loop
```python
for epoch in range(num_epochs):
    for batch_X, batch_y in data_loader:
        # 1. Forward pass
        predictions = model.forward(batch_X)

        # 2. Compute loss
        loss = loss_function(batch_y, predictions)

        # 3. Backward pass
        gradients = model.backward(batch_y)

        # 4. Update weights
        model.update(learning_rate)

        # 5. Track metrics
        accuracy = compute_accuracy(batch_y, predictions)
```

### Learning Rate Guidelines
```
Too high: Loss explodes or oscillates
Too low: Training is very slow
Good starting points:
  - 0.001 (Adam optimizer)
  - 0.01 (SGD)
  - 0.1 (sometimes for simple problems)

Use learning rate decay:
lr = initial_lr * (decay_rate ^ epoch)
```

### Batch Size Guidelines
```
Small batch (1-32):
  + More noise helps escape local minima
  + More frequent updates
  - Slower due to less parallelism
  - Noisy gradient estimates

Large batch (256-1024):
  + More stable gradients
  + Better GPU utilization
  - Might get stuck in sharp minima
  - Needs more epochs

Common: 32, 64, 128
```

---

## 🐛 Common Bugs & Fixes

### Shape Mismatches
```python
# Always print shapes when debugging!
print(f"A shape: {A.shape}, B shape: {B.shape}")

# Common issue: Forgot transpose
# Wrong: gradients = error @ weights
# Right: gradients = error @ weights.T
```

### Exploding Gradients
```
Symptoms: Loss becomes NaN or Infinity
Fixes:
  - Lower learning rate
  - Gradient clipping: clip gradients to [-1, 1]
  - Better weight initialization
  - Batch normalization
```

### Vanishing Gradients
```
Symptoms: Loss stops decreasing, early layers don't learn
Fixes:
  - Use ReLU instead of sigmoid
  - Better weight initialization (Xavier/He)
  - Residual connections
  - Batch normalization
```

### Not Learning (Loss constant)
```
Checklist:
  [ ] Is learning rate too low?
  [ ] Are gradients being computed?
  [ ] Is data normalized?
  [ ] Is model too simple?
  [ ] Are weights updating?
  [ ] Is loss function appropriate?
```

### Overfitting
```
Symptoms: Training accuracy high, test accuracy low
Fixes:
  - More training data
  - Data augmentation
  - Dropout
  - L2 regularization
  - Simpler model
  - Early stopping
```

---

## 🔧 Debugging Checklist

### Before Running
- [ ] Check all shapes match
- [ ] Initialize weights properly
- [ ] Normalize input data
- [ ] Start with small learning rate

### During Training
- [ ] Loss is decreasing
- [ ] Gradients are not NaN/Inf
- [ ] Weights are changing
- [ ] Activations are in reasonable range

### Testing
- [ ] Compare with reference implementation
- [ ] Use gradient checking
- [ ] Test on simple dataset (XOR)
- [ ] Visualize predictions

---

## 💻 Code Snippets

### Weight Initialization
```python
# Xavier/Glorot initialization
import math
limit = math.sqrt(6 / (input_size + output_size))
weights = [[random.uniform(-limit, limit) for _ in range(output_size)]
           for _ in range(input_size)]

# He initialization (for ReLU)
std = math.sqrt(2 / input_size)
weights = [[random.gauss(0, std) for _ in range(output_size)]
           for _ in range(input_size)]
```

### One-Hot Encoding
```python
def one_hot(label, num_classes):
    """Convert label to one-hot vector"""
    vec = [0] * num_classes
    vec[label] = 1
    return vec

# Example: label 3 with 10 classes
# [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

### Data Normalization
```python
# Min-Max normalization to [0, 1]
X_normalized = (X - X.min()) / (X.max() - X.min())

# For images (0-255 pixels)
X_normalized = X / 255.0

# Standardization (mean=0, std=1)
X_standardized = (X - X.mean()) / X.std()
```

---

## 📐 Mathematical Notation Guide

```
Symbol          Meaning
------          -------
x, y            Scalars (single numbers)
x, y            Vectors (1D arrays)
X, Y            Matrices (2D arrays)
W               Weights matrix
b               Bias vector
σ               Sigmoid function
∂               Partial derivative
∇               Gradient (vector of partial derivatives)
Σ               Summation
Π               Product
⊙               Element-wise (Hadamard) product
@, ·            Matrix multiplication
ᵀ               Transpose
ℓ, L            Loss
η               Learning rate (eta)
ε               Small constant (epsilon)
```

---

## 🚀 Performance Tips

### Python
```python
# Slow: Nested loops
for i in range(n):
    for j in range(m):
        result[i][j] = a[i][j] + b[i][j]

# Fast: List comprehension
result = [[a[i][j] + b[i][j] for j in range(m)] for i in range(n)]

# Fastest: NumPy
result = a + b  # Vectorized!
```

### C
```c
// Enable compiler optimizations
gcc -O3 -march=native -ffast-math

// Use restrict keyword for pointers
void multiply(float *restrict A, float *restrict B, float *restrict C)

// Memory alignment
float *aligned = aligned_alloc(64, size * sizeof(float));
```

### CUDA
```cuda
// Use shared memory
__shared__ float tile[TILE_SIZE][TILE_SIZE];

// Coalesced memory access (threads access consecutive memory)
data[threadIdx.x]  // Good!
data[threadIdx.x * stride]  // Bad if stride > 1

// Avoid bank conflicts
// Access same location in shared memory simultaneously

// Minimize host-device transfers
// Transfer once, compute many times
```

---

## 📚 Formula Reference

### Forward Pass
```
Layer l:
z[l] = W[l] @ a[l-1] + b[l]
a[l] = activation(z[l])
```

### Backward Pass
```
Layer l:
dz[l] = da[l] ⊙ activation'(z[l])
dW[l] = (1/m) * dz[l] @ a[l-1]ᵀ
db[l] = (1/m) * sum(dz[l])
da[l-1] = W[l]ᵀ @ dz[l]
```

### Update Rules
```
# Gradient Descent
W = W - η * dW

# Momentum
v = β * v + dW
W = W - η * v

# Adam (simplified)
m = β1 * m + (1-β1) * dW
v = β2 * v + (1-β2) * dW²
W = W - η * m / (√v + ε)
```

---

Keep this file open while coding! Refer back often.

**Pro tip**: Print this as a physical cheat sheet to keep by your computer!
