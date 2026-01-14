import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
import time

# CUDA kernel - Native
matmul_native_kernel_code = """
__global__ void matmul_native_kernel(
    float *A, float *B, float *C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}
"""

# CUDA kernel - Shared Memory (FIXED)
matmul_shared_memory_kernel_code = """
#define TILE_SIZE 16

__global__ void matmul_shared_memory_kernel(
    const float *A, const float *B, float *C,
    int M, int N, int K
) {
    // Shared memory for tiles
    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        // Load tile from A
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < N) {
            A_tile[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        } else {
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from B - FIXED: col < K instead of row < N
        int b_row = t * TILE_SIZE + threadIdx.y;
        if (b_row < N && col < K) {  // ← FIXED!
            B_tile[threadIdx.y][threadIdx.x] = B[b_row * K + col];
        } else {
            B_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += A_tile[threadIdx.y][i] * B_tile[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}
"""

# Compile kernels
mod_native = SourceModule(matmul_native_kernel_code)
matmul_native_kernel = mod_native.get_function("matmul_native_kernel")

mod_shared = SourceModule(matmul_shared_memory_kernel_code)
matmul_shared_kernel = mod_shared.get_function("matmul_shared_memory_kernel")


def matmul_cuda_detailed(A, B, kernel, kernel_name):
    """
    Matrix multiplication with DETAILED timing breakdown
    """
    M, N = A.shape
    N2, K = B.shape
    assert N == N2, "Inner dimensions must match"

    # Allocate result
    C = np.zeros((M, K), dtype=np.float32)

    # Convert to contiguous arrays
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)

    # Launch configuration
    block_size = (16, 16, 1)
    grid_size = (
        (K + block_size[0] - 1) // block_size[0],
        (M + block_size[1] - 1) // block_size[1],
        1
    )

    # === DETAILED TIMING ===

    # Allocate GPU memory
    import pycuda.gpuarray as gpuarray
    start_total = time.time()

    d_A = gpuarray.to_gpu(A)
    d_B = gpuarray.to_gpu(B)
    d_C = gpuarray.zeros((M, K), dtype=np.float32)

    transfer_time = time.time() - start_total

    # Kernel execution ONLY (using CUDA events for precision)
    start_event = drv.Event()
    end_event = drv.Event()

    start_event.record()
    kernel(
        d_A, d_B, d_C,
        np.int32(M), np.int32(N), np.int32(K),
        block=block_size,
        grid=grid_size
    )
    end_event.record()
    end_event.synchronize()

    kernel_time = start_event.time_till(end_event) / 1000.0  # Convert to seconds

    # Transfer result back
    start_back = time.time()
    C = d_C.get()
    transfer_back_time = time.time() - start_back

    total_time = transfer_time + kernel_time + transfer_back_time

    return C, {
        'total': total_time * 1000,
        'transfer_h2d': transfer_time * 1000,
        'kernel': kernel_time * 1000,
        'transfer_d2h': transfer_back_time * 1000,
        'name': kernel_name
    }


# Test
if __name__ == "__main__":
    n = 8192

    print("=" * 70)
    print(f" Matrix Multiplication Benchmark ({n}×{n})")
    print("=" * 70)

    A = np.random.randn(n, n).astype(np.float32)
    B = np.random.randn(n, n).astype(np.float32)

    # NumPy (CPU) - well-optimized baseline
    print("\n1. NumPy (CPU - OpenBLAS/MKL):")
    start = time.time()
    C_numpy = A @ B
    numpy_time = (time.time() - start) * 1000
    print(f"   Time: {numpy_time:.3f} ms")

    # Native CUDA
    print("\n2. CUDA Native (global memory only):")
    C_native, times_native = matmul_cuda_detailed(A, B, matmul_native_kernel, "Native")
    print(f"   Transfer H→D: {times_native['transfer_h2d']:.3f} ms")
    print(f"   Kernel:       {times_native['kernel']:.3f} ms ⚡")
    print(f"   Transfer D→H: {times_native['transfer_d2h']:.3f} ms")
    print(f"   Total:        {times_native['total']:.3f} ms")
    error_native = np.max(np.abs(C_numpy - C_native))
    print(f"   Max error:    {error_native:.6e}")

    # Shared Memory CUDA
    print("\n3. CUDA Tiled (shared memory optimization):")
    C_shared, times_shared = matmul_cuda_detailed(A, B, matmul_shared_kernel, "Shared")
    print(f"   Transfer H→D: {times_shared['transfer_h2d']:.3f} ms")
    print(f"   Kernel:       {times_shared['kernel']:.3f} ms ⚡⚡⚡")
    print(f"   Transfer D→H: {times_shared['transfer_d2h']:.3f} ms")
    print(f"   Total:        {times_shared['total']:.3f} ms")
    error_shared = np.max(np.abs(C_numpy - C_shared))
    print(f"   Max error:    {error_shared:.6e}")

    # Analysis
    print("\n" + "=" * 70)
    print(" PERFORMANCE ANALYSIS")
    print("=" * 70)

    print(f"\nNumPy time:                    {numpy_time:.3f} ms")
    print(f"Native CUDA kernel:            {times_native['kernel']:.3f} ms")
    print(f"Shared CUDA kernel:            {times_shared['kernel']:.3f} ms")

    print(f"\nKernel speedup vs NumPy:")
    print(f"  Native:  {numpy_time / times_native['kernel']:.2f}x")
    print(f"  Shared:  {numpy_time / times_shared['kernel']:.2f}x")

    print(f"\nTiled vs Native speedup:")
    print(f"  {times_native['kernel'] / times_shared['kernel']:.2f}x faster!")

    # Compute performance
    ops = 2.0 * n * n * n  # Multiply-add operations
    native_gflops = (ops / 1e9) / (times_native['kernel'] / 1000)
    shared_gflops = (ops / 1e9) / (times_shared['kernel'] / 1000)
    numpy_gflops = (ops / 1e9) / (numpy_time / 1000)

    print(f"\nCompute performance:")
    print(f"  NumPy:          {numpy_gflops:.2f} GFLOPS")
    print(f"  Native CUDA:    {native_gflops:.2f} GFLOPS")
    print(f"  Tiled CUDA:     {shared_gflops:.2f} GFLOPS")
