#include <cuda_runtime.h>



/**
 * Optimized matrix multiplication using shared memory tiling
 for more details, check out this video: https://www.youtube.com/watch?v=Q3GgbfGTnVc
 * Key optimization: Load data into fast shared memory
 * instead of reading from slow global memory repeatedly
 */
 __global__ void matmul_shared_memory_kernel(const float *A, const float *B, float *C, int M, int N, int K){
    /*
    A : M x N
    B : N x K
    C : M x K
    */

    // Shared memory for tiles (fast, on-chip memory)
    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x ;

    float sum = 0.0f;

    // Loop over tiles
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t=0; t<num_tiles; t++){
        // Load tile from A into shared memory
        int a_col = t* TILE_SIZE + threadIdx.x;
        if (row < M && a_col < N){
            A_tile[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        }
        else{
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

    // Load tile from B into shared memory
        int b_row = t* TILE_SIZE + threadIdx.y;
        if (row < N && b_row < K){
            B_tile[threadIdx.y][threadIdx.x] = B[b_row * K + col];
        }
        else{
            B_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Synchronize to ensure all threads loaded their data
        __syncthreads();

        // Compute partial dot product using shared memory
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += A_tile[threadIdx.y][i] * B_tile[i][threadIdx.x];
        }

        // Synchronize before loading next tile
        __syncthreads();

    }
    // Write result
    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}


/*
Dense layer forward pass on GPU
Y = sigmoid(X @ W + b)

Inputs:
    X: (batch_size, input_size) input features
    W: (input_size, output_size) weight matrix
    b: (output_size,) bias vector
Outputs:
    Y: (batch_size, output_size) output features
*/

__global__ void sigmoid_kernel(float *x, int n){ 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n){
        x[index] = 1.0f / (1.0f + expf(-x[index]));
    }
}

__global__ void add_bias_kernel(float *output, const float *bias, int batch_size, int output_size){

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < batch_size && col < output_size){
        output[row*output_size + col] += bias[col];
    }   
}


void forward_pass_gpu(const float *X,const float *W,const float*b,float *Y, int batch_size, int input_size, int output_size) {
    
    // 1. Calculate the multiplication: W @ X
    matmul_shared_memory_kernel<<<batch_size, output_size>>>(X, W, Y, batch_size, input_size, output_size);

    // 2. Add the bias
    dim3 threads(16,16);
    dim3 num_blocks(
        (output_size + threads.x - 1) / threads.x,
        (batch_size + threads.y - 1) / threads.y
    );
    add_bias_kernel<<<num_blocks, threads>>>(Y, b, batch_size, output_size);

    // 3. Apply the sigmoid activation
    int total_elements = batch_size * output_size;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    sigmoid_kernel<<<num_blocks, threads_per_block>>>(Y, total_elements);
    
}