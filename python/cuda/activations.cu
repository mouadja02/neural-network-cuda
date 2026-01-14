#include <cuda_runtime.h>


/*
We will implement the following activation functions:
- Sigmoid
- ReLU
- Softmax
*/

__global__ void sigmoid_kernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void relu_kernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void softmax_kernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float max_val = -FLT_MAX;
        for (int i = 0; i < n; i++) {
            max_val = fmaxf(max_val, input[i]);
        }
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += expf(input[i] - max_val);
        }
        output[idx] = expf(input[idx] - max_val) / sum;
    }
}
