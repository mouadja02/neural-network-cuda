#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

__global__ void vector_add_kernel(const float *a, const float *b, float *c, int n){
    // Calculate the global thread ID
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Check boundaries
    if (index < n){
        c[index] = a[index] + b[index];
    }
}

void vector_add_gpu(const float *h_a, const float *h_b, float *h_c, int n){
    float *d_a, *d_b, *d_c; // device (GPU) memory

    size_t bytes = n * sizeof(float);

    // 1. Allocate device memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // 2. Copy the input data from the host to the device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // 3. Launch the kernel
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    vector_add_kernel<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, n);

    // 4. Check for kernel errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess){
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(error));
        return;
    }

    // 5. Copy from device memory to host memory
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // 6. Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

// GPU version with DETAILED TIMING
void vector_add_gpu_detailed(const float *h_a, const float *h_b, float *h_c, int n,
                              float *memcpy_time, float *kernel_time) {
    float *d_a, *d_b, *d_c;
    size_t bytes = n * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;

    // Allocate
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // TIME: Memory copy H->D
    cudaEventRecord(start);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    float memcpy_h2d = elapsed;

    // TIME: Kernel execution ONLY
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    cudaEventRecord(start);
    vector_add_kernel<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(kernel_time, start, stop);

    // TIME: Memory copy D->H
    cudaEventRecord(start);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    float memcpy_d2h = elapsed;

    *memcpy_time = memcpy_h2d + memcpy_d2h;

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// CPU version for comparison
void vector_add_cpu(const float *a, const float *b, float *c, int n){
    for (int i = 0; i < n; i++){
        c[i] = a[i] + b[i];
    }
}

int main(){
    // Use smaller size first to avoid WSL memory issues
    const int N = 100000000; // 100 million (not 1 billion)
    const int bytes = N * sizeof(float);

    printf("Vector size: %d elements (%.2f GB)\n", N, bytes / 1e9);

    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c_cpu = (float*)malloc(bytes);
    float *h_c_gpu = (float*)malloc(bytes);

    if (!h_a || !h_b || !h_c_cpu || !h_c_gpu) {
        printf("Failed to allocate host memory!\n");
        return 1;
    }

    // Initialize arrays
    printf("Initializing data...\n");
    for (int i = 0; i < N; i++){
        h_a[i] = (float)i;
        h_b[i] = (float)(2*i);
    }

    // ========================================
    // GPU DETAILED TIMING
    // ========================================
    printf("\n=== GPU Computation (Detailed) ===\n");
    float memcpy_time, kernel_time;
    vector_add_gpu_detailed(h_a, h_b, h_c_gpu, N, &memcpy_time, &kernel_time);

    printf("Memory transfer time: %.3f ms\n", memcpy_time);
    printf("Kernel execution time: %.3f ms ⚡\n", kernel_time);
    printf("Total GPU time: %.3f ms\n", memcpy_time + kernel_time);

    // ========================================
    // CPU TIMING
    // ========================================
    printf("\n=== CPU Computation ===\n");
    clock_t cpu_start = clock();
    vector_add_cpu(h_a, h_b, h_c_cpu, N);
    clock_t cpu_end = clock();
    float cpu_time = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    printf("CPU time: %.3f ms\n", cpu_time);

    // ========================================
    // VERIFY CORRECTNESS
    // ========================================
    printf("\n=== Verification ===\n");
    int errors = 0;
    for (int i = 0; i < N; i++){ // FIXED: was i>N (always false!)
        if (h_c_cpu[i] != h_c_gpu[i]){
            errors++;
            if (errors < 5){
                printf("Mismatch at %d: GPU=%f, CPU=%f\n", i, h_c_gpu[i], h_c_cpu[i]);
            }
        }
    }
    printf("Errors: %d / %d\n", errors, N);

    // ========================================
    // RESULTS
    // ========================================
    printf("\n=== PERFORMANCE ANALYSIS ===\n");
    printf("CPU time:          %.3f ms\n", cpu_time);
    printf("GPU kernel time:   %.3f ms ⚡\n", kernel_time);
    printf("GPU total time:    %.3f ms\n", memcpy_time + kernel_time);
    printf("\n");
    printf("Kernel speedup:    %.2fx (GPU kernel vs CPU)\n", cpu_time / kernel_time);
    printf("Total speedup:     %.2fx (GPU total vs CPU)\n", cpu_time / (memcpy_time + kernel_time));
    printf("\n");
    printf("Memory bandwidth:  %.2f GB/s\n", (3.0f * bytes / 1e9) / (memcpy_time / 1000.0f));
    printf("Compute throughput: %.2f GFLOPS\n", (N / 1e9) / (kernel_time / 1000.0f));

    // ========================================
    // ANALYSIS
    // ========================================
    printf("\n=== WHY IS MEMORY TRANSFER SLOW? ===\n");
    float memcpy_percent = 100.0f * memcpy_time / (memcpy_time + kernel_time);
    printf("Memory transfer is %.1f%% of total GPU time!\n", memcpy_percent);
    printf("\nThis is normal for simple operations like vector addition.\n");
    printf("The kernel is TOO FAST (just one addition per element).\n");
    printf("Memory transfer dominates.\n");
    printf("\nIn neural networks, we do MANY operations per element:\n");
    printf("  - Matrix multiply: O(n²) operations per element\n");
    printf("  - Multiple layers without transfers\n");
    printf("  - Result: Computation dominates, memory transfer is amortized!\n");

    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c_gpu);
    free(h_c_cpu);

    return 0;
}
