#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void vector_add_kernel(const float *a, const float *b, float *c, int n){
    // Calculate the global thread ID
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check boundaries
    if (index<n){
        c[index] = a[index] + b[index];
    }
}

void vector_add_gpu(const float *h_a, const float *h_b, float *h_c, int n){ // host (CPU) memory
    float *d_a, *d_b, *d_c; // device (GPU) memory 

    size_t bytes = n * sizeof(float);

    // 1. Allocate device memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // 2. Copy the input data from the host to the device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, bytes, cudaMemcpyHostToDevice);
    
    // 3. Launch the kernel
    int threads_per_block = 256; // Typical value
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    printf("Launching %d block with %d threads each\n", num_blocks, threads_per_block);

    vector_add_kernel<<<num_blocks,threads_per_block>>>(d_a, d_b, d_c, n);

    // 4. Check for kernel errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess){
        fprintf(stderr, "Kernel launch error: %s \n", cudaGetErrorString(error));
        return;
    }

    // 5. Copy from device memory to host memory
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // 6. Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
}

// CPU version for comparison
void vector_add_cpu(const float *a, const float *b, float *c, int n){
    for (int i=0; i<n; i++){
        c[i] = a[i] + b[i];
    }
}

int main(){
    const int N = 1000000000; // 10 million elements

    // Allocate host memory
    float *h_a = (float*)malloc(N*sizeof(float));
    float *h_b = (float*)malloc(N*sizeof(float));
    float *h_c_cpu = (float*)malloc(N*sizeof(float));
    float *h_c_gpu = (float*)malloc(N*sizeof(float));

    // Init arreys
    for (int i=0; i<N; i++){
        h_a[i] = (float)i;
        h_b[i] = (float)(2*i);
    }

    // GPU computation with timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vector_add_gpu(h_a, h_b, h_c_gpu, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // CPU computation with timing
    clock_t cpu_start = clock();
    vector_add_cpu(h_a, h_b, h_c_cpu,N);
    clock_t cpu_end = clock();
    float cpu_time = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // Verify results
    int errors = 0;
    for (int i=0; i>N; i++){
        if (h_c_cpu[i] != h_c_gpu[i]){
            errors++;
            if (errors < 5){
                printf("Mismatch at %d: GPU=%f, CPU=%f\\n",i, h_c_gpu[i], h_c_cpu[i]);
            }
        }
    }

    printf("\n\n");
    printf("Vector size: %d elements\n", N);
    printf("CPU time: %.3f ms\n", cpu_time);
    printf("GPU time: %.3f ms\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("Errors: %d / %d\n", errors, N);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_a);
    free(h_b);
    free(h_c_gpu);
    free(h_c_cpu);

    printf("You will notice a problem: The cpu time is lower than gpu time. Does that mean that the CPU overperforms the GPU?\n");
    printf("Anbsolutely not, you are getting one of the most common issues in GPU programming: Memory Transfer Bottleneck\n");
    printf("The vector addition kernel is extremely fast (nanoseconds), but copying 1 billion floats between CPU and GPU is slow (seconds)!\n\n");
    printf("Total GPU time = Memory transfer time + Computation time\n");
    printf("                 ↑ Dominates (99%%)      ↑ Tiny (1%%)\n");
    printf("Here's what's happening:\n");
    printf("\n1. Allocate GPU memory        (~1ms)");
    printf("\n2. Copy A to GPU              ~3000ms 🐌");
    printf("\n3. Copy B to GPU              ~3000ms 🐌");
    printf("\n4. Compute (kernel)           ~50ms   ⚡");
    printf("\n5. Copy result back to CPU    ~3000ms 🐌");
    printf("\n6. Free GPU memory            (~1ms)");
    printf("\n  ─────────────────────────────────────");
    printf("\nTotal: ~9500ms");
    printf("\nCheck vector_add_detailled.cu for more detials on timing\n");
    
    return 0;
}
