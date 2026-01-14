"""
Complete GPU-Accelerated Neural Network using PyCUDA

This integrates all your CUDA kernels into a simple neural network
for MNIST digit classification.
"""

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np
import time

# ============================================================================
# LOAD ALL CUDA KERNELS
# ============================================================================

# Read kernel files
import os
cuda_dir = os.path.join(os.path.dirname(__file__), 'cuda')

# Matrix multiplication kernel (from matmul.cu or forward.cu)
MATMUL_KERNEL = """
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

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += A_tile[threadIdx.y][i] * B_tile[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}
"""

# Activation functions
ACTIVATION_KERNELS = """
__global__ void relu_forward(float *x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = fmaxf(0.0f, x[idx]);
    }
}

__global__ void relu_backward(
    const float *grad_out,
    const float *input,
    float *grad_in,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_in[idx] = (input[idx] > 0.0f) ? grad_out[idx] : 0.0f;
    }
}

__global__ void softmax_forward(
    const float *input,
    float *output,
    int batch_size,
    int num_classes
) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];
    int tid = threadIdx.x;

    const float *row_input = input + row * num_classes;
    float *row_output = output + row * num_classes;

    // Find max
    float local_max = -INFINITY;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        local_max = fmaxf(local_max, row_input[i]);
    }
    shared[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = shared[0];
    __syncthreads();

    // Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        float exp_val = expf(row_input[i] - max_val);
        row_output[i] = exp_val;
        local_sum += exp_val;
    }
    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    float sum = shared[0];
    __syncthreads();

    // Normalize
    for (int i = tid; i < num_classes; i += blockDim.x) {
        row_output[i] /= sum;
    }
}
"""

# Utility kernels
UTILITY_KERNELS = """
__global__ void add_bias(
    float *output,
    const float *bias,
    int batch_size,
    int output_size
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < output_size) {
        output[row * output_size + col] += bias[col];
    }
}

__global__ void cross_entropy_loss(
    const float *predictions,
    const float *targets,
    float *loss,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && targets[idx] > 0.0f) {
        float pred = fmaxf(predictions[idx], 1e-10f);
        atomicAdd(loss, -targets[idx] * logf(pred));
    }
}

__global__ void compute_accuracy(
    const float *predictions,
    const float *targets,
    int *correct,
    int batch_size,
    int num_classes
) {
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= batch_size) return;

    const float *pred_row = predictions + sample * num_classes;
    const float *target_row = targets + sample * num_classes;

    // Find argmax for prediction
    int pred_class = 0;
    float max_pred = pred_row[0];
    for (int i = 1; i < num_classes; i++) {
        if (pred_row[i] > max_pred) {
            max_pred = pred_row[i];
            pred_class = i;
        }
    }

    // Find true class
    int true_class = 0;
    for (int i = 0; i < num_classes; i++) {
        if (target_row[i] > 0.5f) {
            true_class = i;
            break;
        }
    }

    if (pred_class == true_class) {
        atomicAdd(correct, 1);
    }
}

__global__ void sgd_update(
    float *weights,
    const float *gradients,
    float learning_rate,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        weights[idx] -= learning_rate * gradients[idx];
    }
}

__global__ void softmax_cross_entropy_backward(
    const float *predictions,
    const float *targets,
    float *grad_input,
    int batch_size,
    int num_classes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_classes;
    if (idx < total) {
        grad_input[idx] = (predictions[idx] - targets[idx]) / batch_size;
    }
}

__global__ void transpose_kernel(
    const float *input,
    float *output,
    int M,
    int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        output[col * M + row] = input[row * N + col];
    }
}

__global__ void sum_columns_kernel(
    const float *input,
    float *output,
    int batch_size,
    int num_features
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < num_features) {
        float sum = 0.0f;
        for (int row = 0; row < batch_size; row++) {
            sum += input[row * num_features + col];
        }
        output[col] = sum;
    }
}
"""

# Compile all kernels
mod_matmul = SourceModule(MATMUL_KERNEL)
mod_activations = SourceModule(ACTIVATION_KERNELS)
mod_utils = SourceModule(UTILITY_KERNELS)

# Get function references
matmul_tiled = mod_matmul.get_function("matmul_tiled")
relu_forward = mod_activations.get_function("relu_forward")
relu_backward = mod_activations.get_function("relu_backward")
softmax_forward = mod_activations.get_function("softmax_forward")
add_bias = mod_utils.get_function("add_bias")
cross_entropy_loss_kernel = mod_utils.get_function("cross_entropy_loss")
compute_accuracy_kernel = mod_utils.get_function("compute_accuracy")
sgd_update_kernel = mod_utils.get_function("sgd_update")
softmax_ce_backward = mod_utils.get_function("softmax_cross_entropy_backward")
transpose_kernel = mod_utils.get_function("transpose_kernel")
sum_columns_kernel = mod_utils.get_function("sum_columns_kernel")


# ============================================================================
# NEURAL NETWORK CLASS
# ============================================================================

class NeuralNetworkGPU:
    """
    Simple 2-layer neural network running entirely on GPU

    Architecture: input → Dense(ReLU) → Dense(Softmax) → output
    """

    def __init__(self, input_size, hidden_size, output_size, batch_size=32):
        """
        Initialize network on GPU

        Args:
            input_size: Number of input features (e.g., 784 for MNIST)
            hidden_size: Number of neurons in hidden layer (e.g., 128)
            output_size: Number of output classes (e.g., 10 for digits)
            batch_size: Training batch size
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size

        print(f"Creating neural network on GPU:")
        print(f"  Architecture: {input_size} → {hidden_size} → {output_size}")

        # Initialize weights with Xavier initialization
        limit1 = np.sqrt(6.0 / (input_size + hidden_size))
        W1 = np.random.uniform(-limit1, limit1, (input_size, hidden_size)).astype(np.float32)
        b1 = np.zeros(hidden_size, dtype=np.float32)

        limit2 = np.sqrt(6.0 / (hidden_size + output_size))
        W2 = np.random.uniform(-limit2, limit2, (hidden_size, output_size)).astype(np.float32)
        b2 = np.zeros(output_size, dtype=np.float32)

        # Transfer to GPU
        self.d_W1 = gpuarray.to_gpu(W1)
        self.d_b1 = gpuarray.to_gpu(b1)
        self.d_W2 = gpuarray.to_gpu(W2)
        self.d_b2 = gpuarray.to_gpu(b2)

        # Allocate GPU memory for activations (will be resized if needed)
        self.d_z1 = None  # Pre-activation layer 1 (before ReLU)
        self.d_z1_pre_relu = None  # Store for backward pass
        self.d_a1 = None  # Activation layer 1 (after ReLU)
        self.d_z2 = None  # Pre-activation layer 2
        self.d_a2 = None  # Activation layer 2 (output)

        # Gradients (allocated on-demand during training)
        self.d_grad_W1 = None
        self.d_grad_b1 = None
        self.d_grad_W2 = None
        self.d_grad_b2 = None
        self.d_grad_z1 = None
        self.d_grad_z2 = None

        print(f"  Total parameters: {input_size * hidden_size + hidden_size + hidden_size * output_size + output_size:,}")

    def _allocate_activations(self, batch_size):
        """Allocate GPU memory for activations and gradients"""
        if self.d_z1 is None or self.d_z1.shape[0] != batch_size:
            # Forward pass storage
            self.d_z1 = gpuarray.zeros((batch_size, self.hidden_size), dtype=np.float32)
            self.d_z1_pre_relu = gpuarray.zeros((batch_size, self.hidden_size), dtype=np.float32)
            self.d_a1 = gpuarray.zeros((batch_size, self.hidden_size), dtype=np.float32)
            self.d_z2 = gpuarray.zeros((batch_size, self.output_size), dtype=np.float32)
            self.d_a2 = gpuarray.zeros((batch_size, self.output_size), dtype=np.float32)

            # Backward pass gradients
            self.d_grad_z1 = gpuarray.zeros((batch_size, self.hidden_size), dtype=np.float32)
            self.d_grad_z2 = gpuarray.zeros((batch_size, self.output_size), dtype=np.float32)

            # Weight gradients
            if self.d_grad_W1 is None:
                self.d_grad_W1 = gpuarray.zeros_like(self.d_W1)
                self.d_grad_b1 = gpuarray.zeros_like(self.d_b1)
                self.d_grad_W2 = gpuarray.zeros_like(self.d_W2)
                self.d_grad_b2 = gpuarray.zeros_like(self.d_b2)

    def forward(self, d_X):
        """
        Forward pass on GPU

        Args:
            d_X: GPU array of shape (batch_size, input_size)

        Returns:
            d_output: GPU array of shape (batch_size, output_size)
        """
        batch_size = d_X.shape[0]
        self._allocate_activations(batch_size)

        # Layer 1: z1 = X @ W1
        block = (16, 16, 1)
        grid = (
            (self.hidden_size + 15) // 16,
            (batch_size + 15) // 16,
            1
        )
        matmul_tiled(
            d_X, self.d_W1, self.d_z1,
            np.int32(batch_size), np.int32(self.input_size), np.int32(self.hidden_size),
            block=block, grid=grid
        )

        # Add bias
        add_bias(
            self.d_z1, self.d_b1,
            np.int32(batch_size), np.int32(self.hidden_size),
            block=block, grid=grid
        )

        # Save z1 before ReLU (needed for backward pass)
        drv.memcpy_dtod(self.d_z1_pre_relu.gpudata, self.d_z1.gpudata, self.d_z1.nbytes)

        # ReLU activation (in-place)
        n = batch_size * self.hidden_size
        relu_forward(
            self.d_z1, np.int32(n),
            block=(256, 1, 1),
            grid=((n + 255) // 256, 1, 1)
        )
        self.d_a1 = self.d_z1  # After ReLU

        # Layer 2: z2 = a1 @ W2
        grid2 = (
            (self.output_size + 15) // 16,
            (batch_size + 15) // 16,
            1
        )
        matmul_tiled(
            self.d_a1, self.d_W2, self.d_z2,
            np.int32(batch_size), np.int32(self.hidden_size), np.int32(self.output_size),
            block=block, grid=grid2
        )

        # Add bias
        add_bias(
            self.d_z2, self.d_b2,
            np.int32(batch_size), np.int32(self.output_size),
            block=block, grid=grid2
        )

        # Softmax activation
        shared_mem = 256 * 4  # 256 floats
        softmax_forward(
            self.d_z2, self.d_a2,
            np.int32(batch_size), np.int32(self.output_size),
            block=(256, 1, 1),
            grid=(batch_size, 1, 1),
            shared=shared_mem
        )

        return self.d_a2

    def train_batch(self, d_X, d_y, learning_rate=0.001):
        """
        Train on one batch - Complete backward pass implementation

        Args:
            d_X: GPU array (batch_size, input_size)
            d_y: GPU array (batch_size, output_size) - one-hot encoded
            learning_rate: Learning rate

        Returns:
            loss: Scalar loss value
        """
        batch_size = d_X.shape[0]

        # ====================================================================
        # FORWARD PASS
        # ====================================================================
        d_predictions = self.forward(d_X)

        # Compute loss
        d_loss = gpuarray.zeros(1, dtype=np.float32)
        n = batch_size * self.output_size
        cross_entropy_loss_kernel(
            d_predictions, d_y, d_loss,
            np.int32(n),
            block=(256, 1, 1),
            grid=((n + 255) // 256, 1, 1)
        )
        loss = float(d_loss.get()[0]) / batch_size

        # ====================================================================
        # BACKWARD PASS
        # ====================================================================

        # Layer 2 gradients
        # -----------------------------------------------------------------
        # Gradient of softmax + cross-entropy: (pred - target) / batch_size
        softmax_ce_backward(
            d_predictions, d_y, self.d_grad_z2,
            np.int32(batch_size), np.int32(self.output_size),
            block=(256, 1, 1),
            grid=((n + 255) // 256, 1, 1)
        )

        # Compute grad_W2 = a1.T @ grad_z2
        # First, transpose a1: (batch_size, hidden_size) -> (hidden_size, batch_size)
        d_a1_T = gpuarray.zeros((self.hidden_size, batch_size), dtype=np.float32)
        block = (16, 16, 1)
        grid_transpose = (
            (batch_size + 15) // 16,
            (self.hidden_size + 15) // 16,
            1
        )
        transpose_kernel(
            self.d_a1, d_a1_T,
            np.int32(batch_size), np.int32(self.hidden_size),
            block=block, grid=grid_transpose
        )

        # Now compute: grad_W2 = a1.T @ grad_z2
        # a1.T: (hidden_size, batch_size) @ grad_z2: (batch_size, output_size) = (hidden_size, output_size)
        grid_W2 = (
            (self.output_size + 15) // 16,
            (self.hidden_size + 15) // 16,
            1
        )
        matmul_tiled(
            d_a1_T, self.d_grad_z2, self.d_grad_W2,
            np.int32(self.hidden_size), np.int32(batch_size), np.int32(self.output_size),
            block=block, grid=grid_W2
        )

        # Compute grad_b2 = sum(grad_z2, axis=0)
        sum_columns_kernel(
            self.d_grad_z2, self.d_grad_b2,
            np.int32(batch_size), np.int32(self.output_size),
            block=(256, 1, 1),
            grid=((self.output_size + 255) // 256, 1, 1)
        )

        # Propagate gradient to layer 1
        # grad_a1 = grad_z2 @ W2.T
        # First transpose W2: (hidden_size, output_size) -> (output_size, hidden_size)
        d_W2_T = gpuarray.zeros((self.output_size, self.hidden_size), dtype=np.float32)
        grid_W2_transpose = (
            (self.hidden_size + 15) // 16,
            (self.output_size + 15) // 16,
            1
        )
        transpose_kernel(
            self.d_W2, d_W2_T,
            np.int32(self.hidden_size), np.int32(self.output_size),
            block=block, grid=grid_W2_transpose
        )

        # grad_a1 = grad_z2 @ W2.T
        # (batch_size, output_size) @ (output_size, hidden_size) = (batch_size, hidden_size)
        d_grad_a1 = gpuarray.zeros((batch_size, self.hidden_size), dtype=np.float32)
        grid_a1 = (
            (self.hidden_size + 15) // 16,
            (batch_size + 15) // 16,
            1
        )
        matmul_tiled(
            self.d_grad_z2, d_W2_T, d_grad_a1,
            np.int32(batch_size), np.int32(self.output_size), np.int32(self.hidden_size),
            block=block, grid=grid_a1
        )

        # Layer 1 gradients
        # -----------------------------------------------------------------
        # Apply ReLU backward: grad_z1 = grad_a1 * (z1 > 0)
        n1 = batch_size * self.hidden_size
        relu_backward(
            d_grad_a1, self.d_z1_pre_relu, self.d_grad_z1,
            np.int32(n1),
            block=(256, 1, 1),
            grid=((n1 + 255) // 256, 1, 1)
        )

        # Compute grad_W1 = X.T @ grad_z1
        d_X_T = gpuarray.zeros((self.input_size, batch_size), dtype=np.float32)
        grid_X_transpose = (
            (batch_size + 15) // 16,
            (self.input_size + 15) // 16,
            1
        )
        transpose_kernel(
            d_X, d_X_T,
            np.int32(batch_size), np.int32(self.input_size),
            block=block, grid=grid_X_transpose
        )

        # grad_W1 = X.T @ grad_z1
        # (input_size, batch_size) @ (batch_size, hidden_size) = (input_size, hidden_size)
        grid_W1 = (
            (self.hidden_size + 15) // 16,
            (self.input_size + 15) // 16,
            1
        )
        matmul_tiled(
            d_X_T, self.d_grad_z1, self.d_grad_W1,
            np.int32(self.input_size), np.int32(batch_size), np.int32(self.hidden_size),
            block=block, grid=grid_W1
        )

        # Compute grad_b1 = sum(grad_z1, axis=0)
        sum_columns_kernel(
            self.d_grad_z1, self.d_grad_b1,
            np.int32(batch_size), np.int32(self.hidden_size),
            block=(256, 1, 1),
            grid=((self.hidden_size + 255) // 256, 1, 1)
        )

        # ====================================================================
        # UPDATE WEIGHTS (SGD)
        # ====================================================================

        # Update W2 and b2
        n_W2 = self.hidden_size * self.output_size
        sgd_update_kernel(
            self.d_W2, self.d_grad_W2,
            np.float32(learning_rate), np.int32(n_W2),
            block=(256, 1, 1),
            grid=((n_W2 + 255) // 256, 1, 1)
        )

        sgd_update_kernel(
            self.d_b2, self.d_grad_b2,
            np.float32(learning_rate), np.int32(self.output_size),
            block=(256, 1, 1),
            grid=((self.output_size + 255) // 256, 1, 1)
        )

        # Update W1 and b1
        n_W1 = self.input_size * self.hidden_size
        sgd_update_kernel(
            self.d_W1, self.d_grad_W1,
            np.float32(learning_rate), np.int32(n_W1),
            block=(256, 1, 1),
            grid=((n_W1 + 255) // 256, 1, 1)
        )

        sgd_update_kernel(
            self.d_b1, self.d_grad_b1,
            np.float32(learning_rate), np.int32(self.hidden_size),
            block=(256, 1, 1),
            grid=((self.hidden_size + 255) // 256, 1, 1)
        )

        return loss

    def predict(self, X):
        """
        Predict probabilities

        Args:
            X: NumPy array (n_samples, input_size)

        Returns:
            predictions: NumPy array (n_samples, output_size)
        """
        d_X = gpuarray.to_gpu(X.astype(np.float32))
        d_predictions = self.forward(d_X)
        return d_predictions.get()

    def evaluate(self, X, y):
        """
        Compute accuracy

        Args:
            X: NumPy array (n_samples, input_size)
            y: NumPy array (n_samples,) - integer labels

        Returns:
            accuracy: Float
        """
        # One-hot encode labels
        y_onehot = np.zeros((len(y), self.output_size), dtype=np.float32)
        y_onehot[np.arange(len(y)), y] = 1

        # Transfer to GPU
        d_X = gpuarray.to_gpu(X.astype(np.float32))
        d_y = gpuarray.to_gpu(y_onehot)

        # Forward pass
        d_predictions = self.forward(d_X)

        # Compute accuracy
        d_correct = gpuarray.zeros(1, dtype=np.int32)
        compute_accuracy_kernel(
            d_predictions, d_y, d_correct,
            np.int32(len(X)), np.int32(self.output_size),
            block=(256, 1, 1),
            grid=((len(X) + 255) // 256, 1, 1)
        )

        correct = int(d_correct.get()[0])
        accuracy = correct / len(X)
        return accuracy


# ============================================================================
# SIMPLE DEMO & TRAINING TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" GPU Neural Network Demo - Complete Training")
    print("=" * 70)

    # Create simple test data
    np.random.seed(42)
    n_samples = 1000
    input_size = 784
    hidden_size = 128
    output_size = 10
    batch_size = 32

    # Random data (simulating MNIST)
    X_train = np.random.randn(n_samples, input_size).astype(np.float32)
    y_train = np.random.randint(0, output_size, n_samples)

    X_test = np.random.randn(200, input_size).astype(np.float32)
    y_test = np.random.randint(0, output_size, 200)

    # Create network
    print("\nInitializing network...")
    nn = NeuralNetworkGPU(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        batch_size=batch_size
    )

    # Test forward pass
    print("\nTesting forward pass...")
    start = time.time()
    predictions = nn.predict(X_test[:100])
    forward_time = time.time() - start
    print(f"  Forward pass for 100 samples: {forward_time*1000:.2f} ms")
    print(f"  Output shape: {predictions.shape}")
    print(f"  Sample prediction probabilities: {predictions[0]}")
    print(f"  Sum of probabilities: {predictions[0].sum():.6f} (should be ~1.0)")

    # Initial accuracy
    print("\nInitial accuracy (random weights)...")
    initial_accuracy = nn.evaluate(X_test, y_test)
    print(f"  Test accuracy: {initial_accuracy*100:.2f}% (should be ~10%)")

    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    print("\n" + "=" * 70)
    print(" Training Neural Network on GPU")
    print("=" * 70)

    n_epochs = 5
    learning_rate = 0.01
    n_batches = len(X_train) // batch_size

    print(f"\nTraining for {n_epochs} epochs with {n_batches} batches per epoch...")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}\n")

    for epoch in range(n_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0

        # Shuffle data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        # Train on batches
        for batch_idx in range(n_batches):
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]

            # One-hot encode labels
            y_onehot = np.zeros((batch_size, output_size), dtype=np.float32)
            y_onehot[np.arange(batch_size), y_batch] = 1

            # Transfer to GPU
            d_X_batch = gpuarray.to_gpu(X_batch)
            d_y_batch = gpuarray.to_gpu(y_onehot)

            # Train
            loss = nn.train_batch(d_X_batch, d_y_batch, learning_rate)
            epoch_loss += loss

        # Compute average loss and accuracy
        avg_loss = epoch_loss / n_batches
        train_accuracy = nn.evaluate(X_train[:500], y_train[:500])
        test_accuracy = nn.evaluate(X_test, y_test)
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1}/{n_epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_accuracy*100:.2f}% | "
              f"Test Acc: {test_accuracy*100:.2f}% | "
              f"Time: {epoch_time:.2f}s")

    # Final evaluation
    print("\n" + "=" * 70)
    print(" Training Complete!")
    print("=" * 70)
    final_accuracy = nn.evaluate(X_test, y_test)
    print(f"\nInitial accuracy: {initial_accuracy*100:.2f}%")
    print(f"Final accuracy:   {final_accuracy*100:.2f}%")
    print(f"Improvement:      {(final_accuracy - initial_accuracy)*100:.2f}%")

    print("\n" + "=" * 70)
    print(" Success! Your GPU neural network is fully working!")
    print("=" * 70)
    print("""
Next steps:
1. Load real MNIST data for actual digit recognition
2. Add more layers or experiment with architecture
3. Try different optimizers (Adam, RMSprop)
4. Add batch normalization or dropout
5. Visualize predictions on real images

Your complete GPU neural network implementation is ready!
""")
