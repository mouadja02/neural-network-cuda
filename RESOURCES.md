# Learning Resources

This document contains all the reference materials for your journey.

---

## 📺 Essential Video Series

### 1. Linear Algebra - 3Blue1Brown
**Watch these FIRST** (15 episodes, ~3 hours total)
- [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
  - Chapter 1: Vectors, what even are they?
  - Chapter 2: Linear combinations, span, and basis vectors
  - Chapter 3: **Linear transformations and matrices** ⭐ (Critical for understanding matrix multiply)
  - Chapter 4: Matrix multiplication as composition
  - Chapter 5: The determinant
  - Chapter 9: Dot products and duality
  - Chapter 10: Cross products
  - Chapter 14: Eigenvectors and eigenvalues

**Why this matters**: Understanding matrices as *transformations* (not just arrays of numbers) is the key insight for neural networks.

### 2. Neural Networks - 3Blue1Brown
**Watch after Exercise 1.5** (4 episodes)
- [Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
  - Chapter 1: But what is a neural network?
  - Chapter 2: Gradient descent, how neural networks learn
  - Chapter 3: **What is backpropagation really doing?** ⭐ (Critical!)
  - Chapter 4: Backpropagation calculus

### 3. Neural Networks: Zero to Hero - Andrej Karpathy
**Watch during Phase 2** (Advanced, each ~2 hours)
- [Building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) - Autograd engine from scratch
- [Building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) - Character-level language model
- [Building GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) - Transformer from scratch

---

## 📚 Books

### Beginner (Start Here)

1. **Deep Learning Book** by Goodfellow, Bengio, Courville
   - [Free online](https://www.deeplearningbook.org/)
   - Read: Chapter 2 (Linear Algebra), Chapter 6 (Deep Feedforward Networks)
   - Dense but authoritative

2. **Neural Networks and Deep Learning** by Michael Nielsen
   - [Free online](http://neuralnetworksanddeeplearning.com/)
   - More beginner-friendly with interactive examples
   - Great intuition building

### Intermediate (Phase 3+)

3. **Programming Massively Parallel Processors** by Kirk & Hwu
   - THE book for learning CUDA
   - Read: Chapters 1-6 for fundamentals
   - Examples in book: https://github.com/tthakore/PMPP-Book-Examples

4. **Mathematics for Machine Learning**
   - [Free PDF](https://mml-book.github.io/)
   - Comprehensive math reference

### Reference

5. **The Matrix Cookbook**
   - [Free PDF](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)
   - Quick reference for matrix derivatives
   - Keep this open during backprop implementation

---

## 🎓 Online Courses

### Phase 1-2: Fundamentals
1. **Khan Academy - Linear Algebra**
   - [Course link](https://www.khanacademy.org/math/linear-algebra)
   - Sections to focus on:
     - Vectors and spaces
     - Matrix transformations
     - Dot and cross products

2. **Khan Academy - Calculus**
   - [Course link](https://www.khanacademy.org/math/calculus-1)
   - Focus on derivatives and chain rule (critical for backprop!)

3. **Stanford CS231n - CNNs for Visual Recognition**
   - [Course page](http://cs231n.stanford.edu/)
   - [Lecture videos](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)
   - Watch: Lectures 2-4 (Image Classification, Loss Functions, Backprop)
   - Their assignments use NumPy - perfect for Phase 2!

### Phase 4-5: CUDA
4. **NVIDIA Deep Learning Institute**
   - [Fundamentals of Accelerated Computing with CUDA C/C++](https://courses.nvidia.com/courses/course-v1:DLI+C-AC-01+V1/)
   - Official NVIDIA course (sometimes free, sometimes $90)
   - Hands-on with actual GPUs

5. **Udacity - Intro to Parallel Programming**
   - [Course link](https://www.udacity.com/course/intro-to-parallel-programming--cs344)
   - Free, uses CUDA
   - Great for understanding GPU architecture

---

## 📄 Papers to Read

### Must Read (In Order)

1. **"A Neural Network in 11 Lines of Python"** - Andrew Trask
   - [Blog post](https://iamtrask.github.io/2015/07/12/basic-python-network/)
   - Read BEFORE starting your implementation
   - Simple 2-layer network, pure NumPy

2. **"Backpropagation Applied to Handwritten Zip Code Recognition"** - LeCun et al. (1989)
   - Historical paper showing neural nets working on digits
   - See where it all started!

3. **"ImageNet Classification with Deep CNNs"** (AlexNet) - Krizhevsky et al. (2012)
   - The paper that started the deep learning revolution
   - See why GPUs matter for deep learning

4. **"Adam: A Method for Stochastic Optimization"** - Kingma & Ba (2015)
   - Modern optimizer (Phase 5 Challenge)
   - [Paper link](https://arxiv.org/abs/1412.6980)

5. **"Batch Normalization"** - Ioffe & Szegedy (2015)
   - Advanced Challenge topic
   - [Paper link](https://arxiv.org/abs/1502.03167)

### Reference Papers

6. **"Gradient-Based Learning Applied to Document Recognition"** - LeCun et al. (1998)
   - Classic MNIST paper
   - [PDF](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)

---

## 🌐 Websites & Tools

### Documentation
- [Python C API](https://docs.python.org/3/extending/extending.html) - Phase 3
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) - Phase 4
- [NumPy Documentation](https://numpy.org/doc/stable/) - Phase 2
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/) - Advanced

### Interactive Learning
- [Tensorflow Playground](https://playground.tensorflow.org/) - Visualize neural networks in browser
- [Neural Network Simulator](https://www.mladdict.com/neural-network-simulator) - Play with architectures
- [Matrix Multiplication Visualizer](http://matrixmultiplication.xyz/) - See how matrix multiply works

### CUDA Resources
- [CUDA by Example](https://developer.nvidia.com/cuda-example) - Code samples
- [CUDA Toolkit Samples](https://github.com/NVIDIA/cuda-samples) - Official examples
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

### Debugging & Profiling
- [Valgrind](https://valgrind.org/) - Memory leak detector (Linux/WSL)
- [Nsight Compute](https://developer.nvidia.com/nsight-compute) - CUDA profiler
- [NVIDIA Visual Profiler](https://developer.nvidia.com/nvidia-visual-profiler) - Alternative profiler

---

## 🔧 Setup & Tools

### Windows + CUDA Setup

1. **Install Visual Studio 2019/2022** (Community Edition)
   - Required for CUDA on Windows
   - Install "Desktop development with C++"

2. **Install CUDA Toolkit**
   - [Download CUDA 12.x](https://developer.nvidia.com/cuda-downloads)
   - Includes nvcc compiler, libraries, samples
   - Verify: `nvcc --version`

3. **Install Python 3.9+**
   - Already have it, but ensure pip is updated
   - `python -m pip install --upgrade pip`

4. **WSL2 (Optional but Recommended)**
   - For Valgrind and Linux tools
   - `wsl --install`
   - Install Ubuntu 22.04
   - CUDA works in WSL2!

### Recommended VS Code Extensions
- Python
- C/C++
- CUDA C++
- Better Comments
- Code Runner

### Python Packages (Install as Needed)
```bash
# Phase 1-2
pip install matplotlib

# Phase 2+
pip install numpy

# Phase 3
pip install pybind11

# Visualization
pip install rich  # For terminal UI
```

---

## 📊 Datasets

### MNIST (Your Main Dataset)
- **Source**: http://yann.lecun.com/exdb/mnist/
- **Easier**: Use `torchvision` or `keras.datasets`
```python
from torchvision import datasets
mnist = datasets.MNIST('./data', download=True)
```

### Alternative Simple Datasets (For Testing)
- **XOR**: Create yourself (4 samples)
- **Circles**: sklearn.datasets.make_circles
- **CIFAR-10**: More advanced (10,000 images)

---

## 🎯 Checkpoint Resources

### Checkpoint 1 (XOR)
- [Understanding XOR Problem](https://medium.com/@jayeshbahire/the-xor-problem-in-neural-networks-50006411840b)
- Why XOR needs hidden layers (linearly inseparable)

### Checkpoint 2 (MNIST NumPy)
- [MNIST from Scratch in NumPy](https://towardsdatascience.com/mnist-from-scratch-in-numpy-b7d7e7e9e8e8)
- Don't copy code! Use for reference only

### Checkpoint 3 (C Implementation)
- [Matrix Multiplication in C](https://www.geeksforgeeks.org/c-program-multiply-two-matrices/)
- [Python C Extensions Tutorial](https://realpython.com/build-python-c-extension-module/)

### Checkpoint 4 (CUDA)
- [Matrix Multiply in CUDA](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/matrixMul)
- Study but implement yourself!

---

## 🤔 When You're Stuck

### Question Checklist
Before asking for help, try:
1. **Print shapes** - 90% of bugs are shape mismatches
2. **Print values** - Check for NaN, Inf, or unexpected ranges
3. **Test small** - Use 2x2 matrices first
4. **Compare with NumPy** - Does your result match?
5. **Read error message** - Especially line numbers

### Debugging Strategies
- **Binary search**: Comment out half your code to find the bug
- **Unit test**: Test each function individually
- **Visualize**: Plot matrices, weights, activations
- **Simplify**: Remove complexity until it works, then add back

### Where to Ask Questions
1. **Me (Claude)**: I'm your teacher - ask anything!
2. **Stack Overflow**: Search first, then ask
3. **r/learnmachinelearning**: Reddit community
4. **CUDA Forums**: https://forums.developer.nvidia.com/

---

## 💡 Pro Tips

### For Matrix Math
- Always verify shapes before operations
- Use small examples (2x2) to check logic
- Remember: (A @ B).T = B.T @ A.T

### For Neural Networks
- Start with small networks (2-3 layers)
- Use small learning rates (0.001-0.1)
- Check gradients with numerical approximation
- Normalize input data (divide by 255 for images)

### For CUDA
- Start with 1D problems (vector add)
- Print from device code (`printf` works!)
- Check for errors: `cudaGetLastError()`
- Profile before optimizing
- Shared memory is your friend

### For C
- Always free what you malloc
- Use Valgrind early
- Check for NULL pointers
- Initialize variables!

---

## 📈 Progress Tracking

### Week by Week
Keep a learning journal. After each week, write:
1. What did I learn?
2. What was hardest?
3. What questions do I still have?
4. What am I proud of?

### Skills Checklist
Mark these off as you master them:

**Phase 1**:
- [ ] Understand matrix multiplication
- [ ] Implement activation functions
- [ ] Compute derivatives
- [ ] Build a neural network layer
- [ ] Train on XOR

**Phase 2**:
- [ ] Use NumPy efficiently
- [ ] Implement mini-batch training
- [ ] Load real datasets
- [ ] Train on MNIST
- [ ] Achieve >90% accuracy

**Phase 3**:
- [ ] Write C matrix operations
- [ ] Manage memory properly (no leaks!)
- [ ] Create Python bindings
- [ ] Match NumPy results

**Phase 4**:
- [ ] Write CUDA kernels
- [ ] Understand GPU memory model
- [ ] Optimize with shared memory
- [ ] Achieve 20x+ speedup

**Phase 5**:
- [ ] Implement backprop on GPU
- [ ] Full training on GPU
- [ ] Create Python API
- [ ] Train MNIST in <30 seconds

---

## 🎉 Celebration Milestones

Take a moment to celebrate when you:
- ✅ Get your first matrix multiply working
- ✅ See your first neural network train
- ✅ Achieve >90% on MNIST
- ✅ Run your first CUDA kernel
- ✅ Beat NumPy performance
- ✅ Complete the final project

**Remember**: Every expert was once a beginner. You're building something real, from scratch. That's incredibly difficult and incredibly valuable. Be patient with yourself!

---

**Update this document** as you find helpful resources. This is YOUR reference guide!
