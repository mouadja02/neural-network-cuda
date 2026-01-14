#include <cuda_runtime.h>

#define TILE_SIZE 16

#include "matmul.cu"

// Adam optimizer - like vector addition but with more math
__global__ void adam_update(float *weights, float *grad, float *m, float *v, int n, float beta1, float beta2, float learning_rate, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Update momentum terms (just vector operations!)
        m[idx] = beta1 * m[idx] + (1-beta1) * grad[idx];
        v[idx] = beta2 * v[idx] + (1-beta2) * grad[idx] * grad[idx];

        // Update weights
        weights[idx] -= learning_rate * m[idx] / (sqrtf(v[idx]) + epsilon);
    }
}