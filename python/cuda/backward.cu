#include <cuda_runtime.h>

#define TILE_SIZE 16

// Use matmul_shared_memory_kernel from matmul.cu
#include "matmul.cu"

/*
        Backward pass: compute gradients and update weights.

        Given the gradient of loss with respect to output (dL/dY),
        compute:
        1. Gradient with respect to weights: dL/dW = X^T @ dL/dY
        2. Gradient with respect to biases: dL/db = sum(dL/dY, axis=0)
        3. Gradient with respect to input: dL/dX = dL/dY @ W^T

        Then update weights:
        W = W - learning_rate * dL/dW
        b = b - learning_rate * dL/db

        Args:
            dL_dY: Gradient from next layer (batch_size, output_size)
            learning_rate: Step size for gradient descent

        Returns:
            dL_dX: Gradient to pass to previous layer (batch_size, input_size)

        Example:
            dL_dY = Matrix([[0.1, 0.2],
                            [0.3, 0.4]])  # Gradient from next layer
            dL_dX = layer.backward(dL_dY, learning_rate=0.01)
*/

__global__ void transpose_kernel(float *X, float *X_T, int M, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N){
        X_T[col * M + row] = X[row * N + col];
    }
}

__global__ void backward_kernel(float *X, float *grad_output, float *grad_W, float *grad_b, int M, int N, int K){   
    // Compute weight gradients: grad_W = X^T @ grad_output
    float *X_T = (float*)malloc(M * N * sizeof(float));
    transpose_kernel<<<(M + TILE_SIZE - 1) / TILE_SIZE, TILE_SIZE>>>(X, X_T, M, N);
    matmul_gpu_shared_memory(X_T, grad_output, grad_W, X.rows, X.cols, grad_output.cols);
    // Compute input gradients: grad_X = grad_output @ W^T
    float *W_T = (float*)malloc(N * K * sizeof(float));
    transpose_kernel<<<(N + TILE_SIZE - 1) / TILE_SIZE, TILE_SIZE>>>(W, W_T, N, K);
    matmul_gpu_shared_memory(grad_output, W_T, grad_X, grad_output.rows, W.cols, X.cols);
    // Compute bias gradients: grad_b = sum(grad_output, axis=0)
    for (int i = 0; i < N; i++) {
        atomicAdd(&grad_b[i], grad_output[i]);
    }
}
