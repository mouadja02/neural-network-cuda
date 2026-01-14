# CUDA Programming Guide for Neural Networks

A practical guide to implementing neural network operations on NVIDIA GPUs using CUDA.

---

## 📚 Table of Contents

1. [CUDA Basics](#cuda-basics)
2. [Matrix Multiplication](#matrix-multiplication)
3. [Activation Functions](#activation-functions)
4. [Backpropagation](#backpropagation)
5. [Optimization Techniques](#optimization-techniques)
6. [Performance Profiling](#performance-profiling)

---

## 🚀 CUDA Basics

### Thread Hierarchy

CUDA organizes threads in a 3-level hierarchy:

```
Grid (entire kernel launch)
├── Block 0
│   ├── Thread 0
│   ├── Thread 1
│   └── ...
├── Block 1
│   └── ...
└── ...
```

**Key Concepts**:
- **Thread**: Single execution unit
- **Block**: Group of threads (up to 1024 threads)
- **Grid**: Collection of blocks

### Thread Indexing

```cuda
// 1D indexing
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D indexing (for matrices)
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

### Memory Hierarchy

From fastest to slowest:

1. **Registers** (per-thread, ~20KB)
2. **Shared Memory** (per-block, ~48KB)
3. **L1/L2 Cache** (automatic)
4. **Global Memory** (device, GBs)
5. **Host Memory** (CPU RAM)

---

## 🔢 Matrix Multiplication

### Naive Implementation

```cuda
__global__ void matmul_naive(float *A, float *B, float *C, int M, int N, int K) {
    // C[M×K] = A[M×N] @ B[N×K]
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
```

**Launch Configuration**:
```cuda
dim3 block(16, 16);
dim3 grid((K + 15) / 16, (M + 15) / 16);
matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
```

**Performance**: ~50 GFLOPS on RTX 3080

---

### Optimized Implementation (Tiled with Shared Memory)

```cuda
#define TILE_SIZE 16

__global__ void matmul_tiled(
    const float *A, const float *B, float *C,
    int M, int N, int K
) {
    // Shared memory for tiles
    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < num_tiles; t++) {
        // Load tile from A into shared memory
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < N) {
            A_tile[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        } else {
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile from B into shared memory
        int b_row = t * TILE_SIZE + threadIdx.y;
        if (b_row < N && col < K) {
            B_tile[threadIdx.y][threadIdx.x] = B[b_row * K + col];
        } else {
            B_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Synchronize to ensure all threads loaded their data
        __syncthreads();
        
        // Compute partial sum using shared memory
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
```

**Why It's Faster**:
1. **Reduced Global Memory Access**: Each element loaded once per tile instead of N times
2. **Memory Coalescing**: Threads in a warp access consecutive memory
3. **Shared Memory**: 100x faster than global memory

**Performance**: ~2000 GFLOPS on RTX 3080 (40x improvement!)

---

## 🎯 Activation Functions

### ReLU

```cuda
__global__ void relu_forward(float *x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = fmaxf(0.0f, x[idx]);
    }
}

__global__ void relu_backward(
    const float *grad_out,
    const float *input,
    float *grad_in,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_in[idx] = (input[idx] > 0.0f) ? grad_out[idx] : 0.0f;
    }
}
```

**Launch**:
```cuda
int n = batch_size * features;
int block_size = 256;
int grid_size = (n + block_size - 1) / block_size;
relu_forward<<<grid_size, block_size>>>(d_x, n);
```

---

### Softmax

Softmax requires reduction operations (max, sum) which are more complex:

```cuda
__global__ void softmax_forward(
    const float *input,
    float *output,
    int batch_size,
    int num_classes
) {
    int row = blockIdx.x;  // One block per sample
    if (row >= batch_size) return;
    
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    
    const float *row_input = input + row * num_classes;
    float *row_output = output + row * num_classes;
    
    // Step 1: Find max (for numerical stability)
    float local_max = -INFINITY;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        local_max = fmaxf(local_max, row_input[i]);
    }
    shared[tid] = local_max;
    __syncthreads();
    
    // Parallel reduction to find global max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = shared[0];
    __syncthreads();
    
    // Step 2: Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        float exp_val = expf(row_input[i] - max_val);
        row_output[i] = exp_val;
        local_sum += exp_val;
    }
    shared[tid] = local_sum;
    __syncthreads();
    
    // Parallel reduction to find sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    float sum = shared[0];
    __syncthreads();
    
    // Step 3: Normalize
    for (int i = tid; i < num_classes; i += blockDim.x) {
        row_output[i] /= sum;
    }
}
```

**Launch**:
```cuda
int shared_mem = 256 * sizeof(float);
softmax_forward<<<batch_size, 256, shared_mem>>>(d_input, d_output, batch_size, num_classes);
```

---

## 🔄 Backpropagation

### Gradient Computation

For a dense layer: `Y = X @ W + b`

Gradients needed:
- `dL/dW = X.T @ dL/dY`
- `dL/db = sum(dL/dY, axis=0)`
- `dL/dX = dL/dY @ W.T`

```cuda
// Gradient w.r.t. weights: dL/dW = X.T @ dL/dY
// Use transpose + matmul kernels

// Gradient w.r.t. bias: dL/db = sum(dL/dY, axis=0)
__global__ void sum_columns_kernel(
    const float *input,
    float *output,
    int batch_size,
    int num_features
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < num_features) {
        float sum = 0.0f;
        for (int row = 0; row < batch_size; row++) {
            sum += input[row * num_features + col];
        }
        output[col] = sum;
    }
}
```

### Softmax + Cross-Entropy Backward

Special case: Combined derivative is simply `predictions - targets`

```cuda
__global__ void softmax_cross_entropy_backward(
    const float *predictions,
    const float *targets,
    float *grad_input,
    int batch_size,
    int num_classes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_classes;
    
    if (idx < total) {
        grad_input[idx] = (predictions[idx] - targets[idx]) / batch_size;
    }
}
```

---

## ⚡ Optimization Techniques

### 1. Memory Coalescing

**Bad** (strided access):
```cuda
// Threads access non-consecutive memory
float val = data[threadIdx.x * stride];
```

**Good** (coalesced access):
```cuda
// Threads access consecutive memory
float val = data[threadIdx.x];
```

### 2. Shared Memory Bank Conflicts

Shared memory is divided into 32 banks. Avoid multiple threads accessing the same bank:

**Bad**:
```cuda
__shared__ float shared[32][32];
float val = shared[threadIdx.x][0];  // All threads access bank 0
```

**Good**:
```cuda
__shared__ float shared[32][33];  // Padding to avoid conflicts
float val = shared[threadIdx.x][threadIdx.y];
```

### 3. Occupancy

Maximize GPU utilization by having enough threads:

```cuda
// Check occupancy
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel);
```

**Guidelines**:
- Use block sizes that are multiples of 32 (warp size)
- Typical block sizes: 128, 256, 512
- Balance shared memory usage vs occupancy

### 4. Kernel Fusion

Combine multiple operations into one kernel to reduce memory transfers:

**Before** (3 kernel launches):
```cuda
matmul<<<...>>>(X, W, Z);
add_bias<<<...>>>(Z, b);
relu<<<...>>>(Z);
```

**After** (1 kernel launch):
```cuda
matmul_bias_relu<<<...>>>(X, W, b, Z);
```

---

## 📊 Performance Profiling

### Using NVIDIA Nsight Compute

```bash
# Profile kernel
ncu --set full -o profile ./program

# Key metrics to check:
# - SM Efficiency (should be >80%)
# - Memory Throughput (% of peak bandwidth)
# - Occupancy (should be >50%)
# - Bank Conflicts (should be 0)
```

### Using nvprof (deprecated but still useful)

```bash
nvprof --print-gpu-trace ./program
```

### Performance Checklist

- [ ] Memory coalescing optimized
- [ ] Shared memory used where beneficial
- [ ] No bank conflicts
- [ ] Occupancy >50%
- [ ] Minimal host-device transfers
- [ ] Kernel fusion applied
- [ ] Proper block/grid dimensions

---

## 🎯 Best Practices

### 1. Error Checking

Always check CUDA errors:

```cuda
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Usage
CUDA_CHECK(cudaMalloc(&d_data, size));
CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
```

### 2. Asynchronous Operations

Overlap computation and memory transfers:

```cuda
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Overlap transfers and computation
cudaMemcpyAsync(d_data1, h_data1, size, cudaMemcpyHostToDevice, stream1);
kernel<<<grid, block, 0, stream2>>>(d_data2);
```

### 3. Pinned Memory

Use pinned (page-locked) memory for faster transfers:

```cuda
float *h_data;
cudaMallocHost(&h_data, size);  // Pinned memory
// ... use h_data ...
cudaFreeHost(h_data);
```

**Speedup**: 2-3x faster transfers than regular malloc

---

## 📚 Resources

### Official Documentation
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

### Courses
- Udacity: Intro to Parallel Programming
- Coursera: GPU Programming Specialization

### Books
- "Programming Massively Parallel Processors" by Kirk & Hwu
- "CUDA by Example" by Sanders & Kandrot

### Tools
- NVIDIA Nsight Compute (profiler)
- NVIDIA Nsight Systems (system-wide profiler)
- cuda-memcheck (memory checker)

---

## 🎓 Common Pitfalls

1. **Forgetting `__syncthreads()`**: Can cause race conditions
2. **Not checking array bounds**: Leads to memory corruption
3. **Using too much shared memory**: Reduces occupancy
4. **Ignoring warp divergence**: Causes performance loss
5. **Too many host-device transfers**: Bottleneck

---

## 🚀 Performance Targets

For RTX 3080 (10,240 CUDA cores, 760 GB/s memory bandwidth):

| Operation | Target Performance |
|-----------|-------------------|
| Matrix Multiply (8192×8192) | >2000 GFLOPS |
| Vector Add (1M elements) | >500 GB/s |
| ReLU (1M elements) | >700 GB/s |
| Softmax (batch=64, classes=10) | <0.1 ms |

---

**Remember**: GPU programming is about maximizing parallelism and minimizing memory transfers!

*"Premature optimization is the root of all evil, but knowing how to optimize is essential."*
