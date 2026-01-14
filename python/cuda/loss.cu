#include <cuda_runtime.h>


/*
We will implement the following loss functions:
- Mean Squared Error
- Binary Cross-Entropy
- Categorical Cross-Entropy
*/

__global__ void mse_kernel(float *input, float *target, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (input[idx] - target[idx]) * (input[idx] - target[idx]); // (y_pred - y_true)^2
    }
}

__global__ void bce_kernel(float *input, float *target, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = -((target[idx] * log(input[idx])) + ((1 - target[idx]) * log(1 - input[idx]))); // -[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
    }
}

__global__ void cce_kernel(float *input, float *target, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = -((target[idx] * log(input[idx])) + ((1 - target[idx]) * log(1 - input[idx]))); // -[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
    }
}


// Derivative of the loss functions
__global__ void mse_derivative_kernel(float *input, float *target, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (input[idx] - target[idx]) * 2; // 2 * (y_pred - y_true)
    }
}

__global__ void bce_derivative_kernel(float *input, float *target, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (input[idx] - target[idx]) / (input[idx] * (1 - input[idx])); // (y_pred - y_true) / (y_pred * (1 - y_pred))
    }
}

__global__ void cce_derivative_kernel(float *input, float *target, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (input[idx] - target[idx]) / (input[idx] * (1 - input[idx])); // (y_pred - y_true) / (y_pred * (1 - y_pred))
    }
}


__global__ void cce_derivative_kernel(float *input, float *target, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (input[idx] - target[idx]) / (input[idx] * (1 - input[idx])); // (y_pred - y_true) / (y_pred * (1 - y_pred))
    }
}
