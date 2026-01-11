# Neural Network From Scratch

Building a complete neural network library from absolute foundations - no black boxes!

## 🎯 Project Goal

Create a **C + CUDA** neural network library with Python bindings that can recognize handwritten digits (0-9). The focus is on **understanding every detail** of how neural networks work, from linear algebra to GPU computing.

## 📚 Learning Materials

- **[LEARNING_GUIDE.md](LEARNING_GUIDE.md)** - Complete curriculum with 18+ weeks of structured exercises
- **[WEEK1_EXERCISE.md](WEEK1_EXERCISE.md)** - Your first task: Build a Matrix class from scratch
- **[RESOURCES.md](RESOURCES.md)** - Videos, books, papers, and reference materials

## 🗺️ Learning Path

### Phase 1: Foundation (Weeks 1-3) - Pure Python
Build everything from scratch using only Python lists:
- Matrix operations (no NumPy!)
- Activation functions & derivatives
- Loss functions
- Progress bar utility
- Your first neural network layer
- **Checkpoint**: Train XOR with 2-layer network

### Phase 2: Optimization (Weeks 4-5) - NumPy
Port to NumPy and add advanced features:
- Vectorized operations
- Mini-batch training
- Data loading pipeline
- **Checkpoint**: MNIST classifier, >90% accuracy

### Phase 3: C Implementation (Weeks 6-8)
Rewrite core operations in C:
- Matrix operations in C
- Python C extensions
- Memory management
- **Checkpoint**: C inference matches Python results

### Phase 4: CUDA Basics (Weeks 9-11)
Learn GPU programming:
- CUDA kernels
- Memory management
- Optimized matrix multiplication
- **Checkpoint**: 20-50x speedup on forward pass

### Phase 5: Complete NN (Weeks 12-16)
Full training on GPU:
- Backward pass on GPU
- Activation function kernels
- SGD optimizer
- Python API wrapper
- **Checkpoint**: Complete MNIST classifier, >97%, <30s training

### Phase 6: Visualization (Weeks 17-18)
Tools for understanding:
- Network visualizer
- Live training dashboard
- Gradient checker
- Debug utilities

## 🚀 Quick Start

### Week 1 - Your First Exercise

1. **Read the guide**:
   ```bash
   # Open these files in your editor
   LEARNING_GUIDE.md      # Full curriculum
   WEEK1_EXERCISE.md      # Current assignment
   RESOURCES.md           # Reference materials
   ```

2. **Implement Matrix class**:
   ```bash
   # Edit this file
   python/core/matrix.py
   ```

3. **Run tests**:
   ```bash
   cd python/tests
   python test_matrix.py
   ```

4. **Get feedback**: Show your code when done!

## 📁 Project Structure

```
NeuralNetwork-foundations/
├── python/              # Python implementations
│   ├── core/           # Core NN components
│   ├── utils/          # Utilities (progress bar, data loader)
│   ├── viz/            # Visualization tools
│   ├── tests/          # Unit tests
│   └── examples/       # Demo scripts
├── c/                  # C implementations
│   ├── matrix.c/.h     # Matrix operations
│   ├── python_bindings/# Python C extensions
│   └── tests/
├── cuda/               # CUDA kernels
│   ├── matmul.cu       # Matrix multiply
│   ├── activations.cu  # Activation functions
│   ├── backward.cu     # Backpropagation
│   └── train.cu        # Training loop
├── LEARNING_GUIDE.md   # Main curriculum
├── RESOURCES.md        # Learning resources
└── README.md           # This file
```

## 🛠️ Requirements

### Current (Phase 1-2)
- Python 3.9+
- Text editor / IDE

### Later (Phase 2+)
- NumPy

### Later (Phase 3+)
- GCC or MSVC
- Python development headers

### Later (Phase 4+)
- NVIDIA GPU (RTX 3080)
- CUDA Toolkit 12.x
- Visual Studio (Windows)

## 📖 How to Use This Repo

This is a **learning journey**, not a finished product. You will:

1. **Read** the exercise descriptions
2. **Implement** the code yourself
3. **Test** your implementations
4. **Iterate** until tests pass
5. **Understand** why it works
6. **Move forward** to next exercise

**Don't skip ahead!** Each exercise builds on previous ones.

## 💡 Learning Philosophy

- ✅ **Implement everything yourself** - No copy/paste
- ✅ **Test everything** - Every component must be verifiable
- ✅ **Understand the math** - Know why, not just how
- ✅ **Iterate** - Start simple, add complexity gradually
- ✅ **Debug visually** - Build tools to see what's happening
- ✅ **Ask questions** - No question is too basic

## 🎓 Teaching Approach

I'm your teacher in this journey. For each exercise, I provide:

- **Clear goals** - What you're building
- **Detailed instructions** - Step-by-step guidance
- **Test cases** - Verify your implementation
- **Hints** - When you're stuck
- **Explanations** - Why things work this way
- **Resources** - Where to learn more
- **Feedback** - Review your code

## 📊 Progress Tracking

Track your progress by checking off completed exercises:

- [ ] Exercise 1.1: Matrix Operations
- [ ] Exercise 1.2: Activation Functions
- [ ] Exercise 1.3: Loss Functions
- [ ] Exercise 1.4: Progress Bar
- [ ] Exercise 1.5: First Neural Network Layer
- [ ] Checkpoint 1: XOR Network
- [ ] Exercise 2.1: Port to NumPy
- [ ] Exercise 2.2: Mini-Batch Training
- [ ] ... (see LEARNING_GUIDE.md for complete list)

## 🎯 Current Focus

**Week 1, Exercise 1.1**: Implement Matrix class from scratch

**Files to work on**:
- `python/core/matrix.py` - Your implementation
- `python/tests/test_matrix.py` - Test cases

**Goal**: Understand matrix operations by implementing them yourself

## 🆘 Getting Help

When stuck:
1. Read error messages carefully
2. Print variable shapes and values
3. Test with small examples (2x2 matrices)
4. Check the RESOURCES.md for references
5. Show your code and ask questions!

## 🌟 Why This Approach?

Building from scratch teaches you:
- **Deep understanding** - No black boxes
- **Debugging skills** - When things go wrong, you know why
- **Appreciation** - Understand what libraries do for you
- **Fundamentals** - Applicable to any framework
- **Confidence** - You built a neural network!

## 📝 License

This is a personal learning project. Use it however helps you learn!

## 🚦 Next Steps

1. Open [WEEK1_EXERCISE.md](WEEK1_EXERCISE.md)
2. Start implementing `python/core/matrix.py`
3. Run tests with `python python/tests/test_matrix.py`
4. Show your code for feedback!

---

**Let's build something amazing together! 🚀**

*"I hear and I forget. I see and I remember. I do and I understand." - Confucius*
