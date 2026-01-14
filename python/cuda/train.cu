#include <cuda_runtime.h>

#define TILE_SIZE 16

#include "matmul.cu"

struct NeuralNetwork {
    float *W1, *b1, *W2, *b2;
    float *m1, *v1, *m2, *v2;
};

__global__ void train_batch_gpu(NeuralNetwork *nn, float *input, float *target) {
    // 1. Forward pass
    matmul_tiled(input, nn->W1, z1);
    add_bias(z1, nn->b1);
    relu_forward(z1, a1);

    matmul_tiled(a1, nn->W2, z2);
    add_bias(z2, nn->b2);
    softmax_forward(z2, a2);

    // 2. Compute loss
    float loss = cross_entropy_loss(a2, target);

    // 3. Backward pass
    softmax_crossentropy_backward(a2, target, dL_dz2);
    matmul_tiled(dL_dz2, nn->W2.T(), dL_da1);
    relu_backward(dL_da1, a1);
    matmul_tiled(dL_da1, nn->W1.T(), dL_dz1);

    // 4. Update weights
    adam_update(nn->W1, dL_dz1, nn->m1, nn->v1, nn->W1.size, 0.9, 0.999, 0.01, 1e-8);
}