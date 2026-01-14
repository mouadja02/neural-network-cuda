# define TILE_SIZE 16

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

__global__ void matmul_native_kernel(const float *A, const float *B, float *C, int M, int N, int K){
    /*
    A : M x N
    B : N x K
    C : M x K
    */
    // Calculate the global thread ID
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x ;

    // Check boundaries
    if (row < M && col < K){
        float sum = 0.0f;
        for (int i=0; i<N; i++){
            sum += A[row * N + i] * B[i * K + col];
        }

        C[row * K + col] = sum;
    }
}



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

/**
 * Host function to launch matrix multiplication
 */
void matmul_gpu_native(const float *h_A,const float *h_B, float *h_C,int M, int N, int K){
    float *d_A, *d_B, *d_C; // device (GPU) memory
    
    // 1. Allocate device memory
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * K * sizeof(float));
    cudaMalloc(&d_C, M * K * sizeof(float));

    // 2. Copy the input data from the host to the device
    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * K * sizeof(float), cudaMemcpyHostToDevice);

    // 3. Launch the kernel
    dim3 threads_per_block(16,16);
    dim3 num_blocks(
        (K + threads_per_block.x - 1) / threads_per_block.x,
        (M + threads_per_block.y - 1) / threads_per_block.y
    );
    matmul_native_kernel<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, M, N, K);

    // 4. Check for kernel errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess){
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(error));
        return;
    }

    // 5. Copy from device memory to host memory
    cudaMemcpy(h_C, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    // 6. Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

void matmul_gpu_shared_memory(const float *h_A,const float *h_B, float *h_C,int M, int N, int K){
    float *d_A, *d_B, *d_C; // device (GPU) memory
    
    // 1. Allocate device memory
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * K * sizeof(float));
    cudaMalloc(&d_C, M * K * sizeof(float));

    // 2. Copy the input data from the host to the device
    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * K * sizeof(float), cudaMemcpyHostToDevice);

    // 3. Launch the kernel
    dim3 threads_per_block(16,16);
    dim3 num_blocks(
        (K + threads_per_block.x - 1) / threads_per_block.x,
        (M + threads_per_block.y - 1) / threads_per_block.y
    );
    matmul_shared_memory_kernel<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, M, N, K);

    // 4. Check for kernel errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess){
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(error));
        return;
    }

    // 5. Copy from device memory to host memory
    cudaMemcpy(h_C, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    // 6. Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}       

/**
 * CPU matrix multiplication for verification
 */
void matmul_cpu(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int row=0; row<M; row++){
        for (int col=0; col<K; col++){
            float sum = 0.0f;
            for (int n=0; n<N; n++){
                sum += A[row * N + n] * B[n * K + col];
            }
            C[row * K + col] = sum;
        }
    }
}

int main(){
    const int M = 1024, N = 1024, K = 1024;

    // Allocate host memory
    float *h_A = (float*)malloc(M * N *sizeof(float));
    float *h_B = (float*)malloc(N * K *sizeof(float));
    float *h_C_cpu = (float*)malloc(M * K *sizeof(float));
    float *h_C_gpu = (float*)malloc(M * K *sizeof(float));

    // Initialize matrices
    for (int i = 0; i < M * N; i++) h_A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < N * K; i++) h_B[i] = (float)(rand() % 100) / 10.0f;

    printf("Matrix dimensions: (%d×%d) × (%d×%d) = (%d×%d)\n\n",
           M, N, N, K, M, K);

    // GPU computation native
    cudaEvent_t start_native, stop_native;
    cudaEventCreate(&start_native);
    cudaEventCreate(&stop_native);

    cudaEventRecord(start_native);
    matmul_gpu_native(h_A, h_B, h_C_gpu, M, N, K);
    cudaEventRecord(stop_native);
    cudaEventSynchronize(stop_native);

    float gpu_time_native;
    cudaEventElapsedTime(&gpu_time_native, start_native, stop_native);

    // GPU computation shared memory
    cudaEvent_t start_shared, stop_shared;
    cudaEventCreate(&start_shared);
    cudaEventCreate(&stop_shared);

    cudaEventRecord(start_shared);
    matmul_gpu_shared_memory(h_A, h_B, h_C_gpu, M, N, K);
    cudaEventRecord(stop_shared);
    cudaEventSynchronize(stop_shared);

    float gpu_time_shared;
    cudaEventElapsedTime(&gpu_time_shared, start_shared, stop_shared);
    

    // CPU computation
    clock_t cpu_start = clock();
    matmul_cpu(h_A, h_B, h_C_cpu, M, N, K);
    clock_t cpu_end = clock();
    float cpu_time = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // Verify correctness
    float max_error = 0.0f;
    for (int i = 0; i < M * K; i++) {
        float error = fabsf(h_C_gpu[i] - h_C_cpu[i]);
        if (error > max_error) max_error = error;
    }

    printf("CPU time: %.3f ms\n", cpu_time);
    printf("GPU time native: %.3f ms\n", gpu_time_native);
    printf("GPU time shared memory: %.3f ms\n", gpu_time_shared);
    printf("Speedup native: %.2fx\n", cpu_time / gpu_time_native);
    printf("Speedup shared memory: %.2fx\n", cpu_time / gpu_time_shared);
    printf("Max error: %e\n", max_error);
    printf("Performance native: %.2f GFLOPS\n",
           2.0 * M * N * K / (gpu_time_native * 1e6));
    printf("Performance shared memory: %.2f GFLOPS\n",
           2.0 * M * N * K / (gpu_time_shared * 1e6));

    // Cleanup
    cudaEventDestroy(start_native);
    cudaEventDestroy(stop_native);
    cudaEventDestroy(start_shared);
    cudaEventDestroy(stop_shared);
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);

    return 0;
}

/*
Results:
CPU time: 3830.593 ms
GPU time native: 118.935 ms
GPU time shared memory: 4.215 ms
Speedup native: 32.21x
Speedup shared memory: 908.74x
Max error: 1.953125e-03
Performance native: 18.06 GFLOPS
Performance shared memory: 509.45 GFLOPS
--------------------------------------------
We can see that the shared memory version is much faster than the native version.
*/