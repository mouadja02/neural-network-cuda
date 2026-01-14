#include <stdio.h>
#include <cuda_runtime.h>

/**
 * Your first CUDA kernel!
 *
 * __global__ means this runs on the GPU
 * It's called by CPU, executed on GPU
 */
__global__ void hello_kernel() {
    // Every thread has unique IDs
    int thread_id = threadIdx.x;           // Thread within block
    int block_id = blockIdx.x;             // Block within grid
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    printf("Hello from block %d, thread %d (global ID %d)\n",
           block_id, thread_id, global_id);
}

int main() {
    printf("Launching kernel with 2 blocks, 4 threads each...\n\n");

    // Launch kernel: <<<num_blocks, threads_per_block>>>
    hello_kernel<<<2, 4>>>();

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("\\nKernel completed successfully!\\n");
    return 0;
}