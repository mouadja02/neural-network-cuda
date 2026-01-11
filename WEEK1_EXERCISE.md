# Week 1 - Exercise 1.1: Build Your Matrix Class

## 🎯 Goal
Implement a Matrix class from scratch WITHOUT using NumPy. This will teach you:
- How 2D data is stored in memory (row-major order)
- Why matrix multiplication is O(n³)
- The math behind neural network operations

## 📝 Your Task

Create the file `python/core/matrix.py` and implement the following class:

```python
class Matrix:
    """
    A simple matrix implementation using nested Python lists.

    Internal representation: self.data is a list of rows
    Example: [[1, 2, 3],
              [4, 5, 6]]  <- 2 rows, 3 columns
    """

    def __init__(self, data):
        """
        Initialize matrix from 2D list.

        Args:
            data: List of lists, e.g., [[1,2], [3,4]]

        Raises:
            ValueError: If rows have different lengths
        """
        # TODO: Implement this
        # Hints:
        # 1. Check if data is valid (all rows same length)
        # 2. Store data as self.data
        # 3. Store dimensions as self.rows and self.cols
        pass

    def shape(self):
        """Return tuple (rows, cols)"""
        # TODO: Implement
        pass

    def __add__(self, other):
        """
        Add two matrices element-wise.

        Example:
            [[1, 2]]  +  [[5, 6]]  =  [[6, 8]]
            [[3, 4]]     [[7, 8]]     [[10, 12]]

        Args:
            other: Another Matrix object

        Returns:
            New Matrix with element-wise sum

        Raises:
            ValueError: If shapes don't match
        """
        # TODO: Implement
        # Hints:
        # 1. Check shapes match
        # 2. Create new result matrix
        # 3. Add corresponding elements
        pass

    def __sub__(self, other):
        """Subtract two matrices element-wise."""
        # TODO: Similar to __add__
        pass

    def __mul__(self, other):
        """
        Element-wise multiplication (Hadamard product).
        NOT matrix multiplication!

        Example:
            [[1, 2]]  *  [[5, 6]]  =  [[5, 12]]
            [[3, 4]]     [[7, 8]]     [[21, 32]]
        """
        # TODO: Implement
        pass

    def dot(self, other):
        """
        Matrix multiplication: self @ other

        Example:
            [[1, 2]]  @  [[5, 6]]  =  [[1*5+2*7, 1*6+2*8]]  =  [[19, 22]]
            [[3, 4]]     [[7, 8]]     [[3*5+4*7, 3*6+4*8]]     [[43, 50]]

        Rule: (m x n) @ (n x p) = (m x p)

        Args:
            other: Another Matrix (self.cols must equal other.rows)

        Returns:
            New Matrix with shape (self.rows, other.cols)

        Raises:
            ValueError: If dimensions incompatible
        """
        # TODO: Implement the triple nested loop
        # for i in range(self.rows):
        #     for j in range(other.cols):
        #         sum = 0
        #         for k in range(self.cols):
        #             sum += self.data[i][k] * other.data[k][j]
        #         result[i][j] = sum
        pass

    def T(self):
        """
        Transpose: flip rows and columns.

        Example:
            [[1, 2, 3]]   ->   [[1, 4]]
            [[4, 5, 6]]        [[2, 5]]
                               [[3, 6]]

        Returns:
            New Matrix with shape (cols, rows)
        """
        # TODO: Implement
        # Hint: result[j][i] = self.data[i][j]
        pass

    def apply(self, func):
        """
        Apply a function to every element.

        Args:
            func: Function that takes a number and returns a number

        Returns:
            New Matrix with func applied to each element

        Example:
            matrix.apply(lambda x: x * 2)  # Doubles every element
        """
        # TODO: Implement
        pass

    def __repr__(self):
        """String representation for debugging."""
        # TODO: Make it print nicely
        # Example output:
        # Matrix([[1, 2],
        #         [3, 4]])
        pass

    def __eq__(self, other):
        """Check if two matrices are equal."""
        # TODO: Implement
        # Useful for testing!
        pass


def zeros(rows, cols):
    """Create a matrix filled with zeros."""
    # TODO: Return a Matrix object with all zeros
    pass


def ones(rows, cols):
    """Create a matrix filled with ones."""
    # TODO: Return a Matrix object with all ones
    pass


def identity(n):
    """
    Create an identity matrix (ones on diagonal, zeros elsewhere).

    Example for n=3:
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]]
    """
    # TODO: Implement
    pass
```

## ✅ Test Cases

Create `python/tests/test_matrix.py`:

```python
import sys
sys.path.append('..')  # Add parent directory to path

from core.matrix import Matrix, zeros, ones, identity


def test_creation():
    """Test basic matrix creation"""
    m = Matrix([[1, 2, 3], [4, 5, 6]])
    assert m.shape() == (2, 3), "Shape should be (2, 3)"
    print("✓ Creation test passed")


def test_addition():
    """Test matrix addition"""
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[5, 6], [7, 8]])
    c = a + b

    assert c.data == [[6, 8], [10, 12]], "Addition failed"
    print("✓ Addition test passed")


def test_subtraction():
    """Test matrix subtraction"""
    a = Matrix([[5, 7], [9, 11]])
    b = Matrix([[1, 2], [3, 4]])
    c = a - b

    assert c.data == [[4, 5], [6, 7]], "Subtraction failed"
    print("✓ Subtraction test passed")


def test_element_wise_multiply():
    """Test element-wise multiplication"""
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[2, 3], [4, 5]])
    c = a * b

    assert c.data == [[2, 6], [12, 20]], "Element-wise multiply failed"
    print("✓ Element-wise multiplication test passed")


def test_dot_product():
    """Test matrix multiplication"""
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[5, 6], [7, 8]])
    c = a.dot(b)

    # [[1*5 + 2*7, 1*6 + 2*8],     [[19, 22],
    #  [3*5 + 4*7, 3*6 + 4*8]]  =   [43, 50]]

    assert c.shape() == (2, 2), "Dot product shape wrong"
    assert c.data == [[19, 22], [43, 50]], "Dot product values wrong"
    print("✓ Dot product test passed")


def test_transpose():
    """Test matrix transpose"""
    a = Matrix([[1, 2, 3], [4, 5, 6]])
    b = a.T()

    assert b.shape() == (3, 2), "Transpose shape wrong"
    assert b.data == [[1, 4], [2, 5], [3, 6]], "Transpose values wrong"
    print("✓ Transpose test passed")


def test_apply():
    """Test apply function"""
    a = Matrix([[1, 2], [3, 4]])
    b = a.apply(lambda x: x * 2)

    assert b.data == [[2, 4], [6, 8]], "Apply failed"
    print("✓ Apply test passed")


def test_special_matrices():
    """Test zeros, ones, identity"""
    z = zeros(2, 3)
    assert z.data == [[0, 0, 0], [0, 0, 0]], "Zeros failed"

    o = ones(2, 2)
    assert o.data == [[1, 1], [1, 1]], "Ones failed"

    i = identity(3)
    assert i.data == [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "Identity failed"

    print("✓ Special matrices test passed")


def test_non_square_multiply():
    """Test multiplication with non-square matrices"""
    # (2x3) @ (3x2) = (2x2)
    a = Matrix([[1, 2, 3],
                [4, 5, 6]])
    b = Matrix([[7, 8],
                [9, 10],
                [11, 12]])
    c = a.dot(b)

    assert c.shape() == (2, 2), "Shape should be (2, 2)"
    # [[1*7+2*9+3*11, 1*8+2*10+3*12],     [[58, 64],
    #  [4*7+5*9+6*11, 4*8+5*10+6*12]]  =   [139, 154]]
    assert c.data == [[58, 64], [139, 154]], "Non-square multiply failed"
    print("✓ Non-square multiplication test passed")


def test_identity_property():
    """Test that A @ I = A"""
    a = Matrix([[1, 2], [3, 4]])
    i = identity(2)
    b = a.dot(i)

    assert b.data == a.data, "Identity property failed"
    print("✓ Identity property test passed")


def run_all_tests():
    """Run all test cases"""
    print("\n" + "="*50)
    print("Running Matrix Tests")
    print("="*50 + "\n")

    test_creation()
    test_addition()
    test_subtraction()
    test_element_wise_multiply()
    test_dot_product()
    test_transpose()
    test_apply()
    test_special_matrices()
    test_non_square_multiply()
    test_identity_property()

    print("\n" + "="*50)
    print("ALL TESTS PASSED! 🎉")
    print("="*50 + "\n")


if __name__ == "__main__":
    run_all_tests()
```

## 🚀 How to Work on This

1. **Start with `__init__`**: Get the basic structure right
2. **Implement `shape()` and `__repr__()`**: These help with debugging
3. **Do addition first**: It's the simplest operation
4. **Then subtraction and element-wise multiply**: Similar patterns
5. **Tackle `dot()` last**: This is the hardest (triple nested loop)
6. **Test as you go**: Run tests after each method

## 🧪 Running Tests

```bash
cd python/tests
python test_matrix.py
```

You should see:
```
==================================================
Running Matrix Tests
==================================================

✓ Creation test passed
✓ Addition test passed
✓ Subtraction test passed
...
✓ Identity property test passed

==================================================
ALL TESTS PASSED! 🎉
==================================================
```

## 💡 Hints

### For `dot()` (matrix multiplication):
The key insight: Element (i,j) in result is the dot product of row i from first matrix and column j from second matrix.

```
For C = A @ B:
C[i][j] = sum of A[i][k] * B[k][j] for all k
```

Example:
```
[[1, 2]]  @  [[5]]  =  [[ 1*5 + 2*6 ]]  =  [[17]]
             [[6]]
```

### For `T()` (transpose):
```
If A is (2x3):        If A.T() is (3x2):
[[1, 2, 3]            [[1, 4]
 [4, 5, 6]]            [2, 5]
                       [3, 6]]

Pattern: B[j][i] = A[i][j]
```

## ❓ Understanding Questions

Before you start coding, make sure you understand:

1. **Why does matrix multiplication require self.cols == other.rows?**
2. **What's the difference between `*` (element-wise) and `dot()` (matrix multiply)?**
3. **Why is transpose important for neural networks?** (Hint: think about backpropagation)
4. **What's the computational complexity of matrix multiplication?** (How many operations for NxN matrices?)

## 🎯 Success Criteria

- All 10 tests pass
- You can explain why matrix multiplication is O(n³)
- You understand the difference between element-wise and matrix operations
- Code is clean and readable

## 📤 What to Do When Done

1. Run all tests and make sure they pass
2. Try multiplying two 100x100 matrices and see how long it takes
3. Show me your code
4. Answer the understanding questions above

Then I'll give you feedback and we'll move to Exercise 1.2: Activation Functions!

## 🐛 Common Mistakes to Avoid

- **Forgetting to check shapes** before operations
- **Confusing element-wise multiply with matrix multiply**
- **Getting loop indices wrong** in `dot()` (i, j, k order matters!)
- **Modifying self.data** instead of creating a new matrix
- **Not handling edge cases** (1x1 matrices, non-square matrices)

---

**Ready? Start coding! Remember: It's okay to get stuck. That's how learning happens. Show me your code when you need help! 🚀**
