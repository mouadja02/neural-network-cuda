"""
Test cases for Matrix class

Run this file to test your implementation:
    cd python/tests
    python test_matrix.py

All tests should pass before moving to the next exercise!
"""

import sys
import os

# Add parent directory to path so we can import from core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.matrix import Matrix, zeros, ones, identity


def test_creation():
    """Test basic matrix creation"""
    print("Testing matrix creation...", end=" ")

    m = Matrix([[1, 2, 3], [4, 5, 6]])
    assert m.shape() == (2, 3), f"Shape should be (2, 3), got {m.shape()}"

    print("✓ PASSED")


def test_addition():
    """Test matrix addition"""
    print("Testing matrix addition...", end=" ")

    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[5, 6], [7, 8]])
    c = a + b

    expected = [[6, 8], [10, 12]]
    assert c.data == expected, f"Expected {expected}, got {c.data}"

    print("✓ PASSED")


def test_subtraction():
    """Test matrix subtraction"""
    print("Testing matrix subtraction...", end=" ")

    a = Matrix([[5, 7], [9, 11]])
    b = Matrix([[1, 2], [3, 4]])
    c = a - b

    expected = [[4, 5], [6, 7]]
    assert c.data == expected, f"Expected {expected}, got {c.data}"

    print("✓ PASSED")


def test_element_wise_multiply():
    """Test element-wise multiplication"""
    print("Testing element-wise multiplication...", end=" ")

    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[2, 3], [4, 5]])
    c = a * b

    expected = [[2, 6], [12, 20]]
    assert c.data == expected, f"Expected {expected}, got {c.data}"

    print("✓ PASSED")


def test_dot_product():
    """Test matrix multiplication"""
    print("Testing matrix multiplication...", end=" ")

    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[5, 6], [7, 8]])
    c = a.dot(b)

    # Manual calculation:
    # [[1*5 + 2*7, 1*6 + 2*8],     [[19, 22],
    #  [3*5 + 4*7, 3*6 + 4*8]]  =   [43, 50]]

    assert c.shape() == (2, 2), f"Shape should be (2, 2), got {c.shape()}"
    expected = [[19, 22], [43, 50]]
    assert c.data == expected, f"Expected {expected}, got {c.data}"

    print("✓ PASSED")


def test_transpose():
    """Test matrix transpose"""
    print("Testing transpose...", end=" ")

    a = Matrix([[1, 2, 3], [4, 5, 6]])
    b = a.T()

    assert b.shape() == (3, 2), f"Shape should be (3, 2), got {b.shape()}"
    expected = [[1, 4], [2, 5], [3, 6]]
    assert b.data == expected, f"Expected {expected}, got {b.data}"

    print("✓ PASSED")


def test_apply():
    """Test apply function"""
    print("Testing apply function...", end=" ")

    a = Matrix([[1, 2], [3, 4]])
    b = a.apply(lambda x: x * 2)

    expected = [[2, 4], [6, 8]]
    assert b.data == expected, f"Expected {expected}, got {b.data}"

    print("✓ PASSED")


def test_special_matrices():
    """Test zeros, ones, identity"""
    print("Testing special matrices...", end=" ")

    z = zeros(2, 3)
    expected_zeros = [[0, 0, 0], [0, 0, 0]]
    assert z.data == expected_zeros, f"Zeros failed: expected {expected_zeros}, got {z.data}"

    o = ones(2, 2)
    expected_ones = [[1, 1], [1, 1]]
    assert o.data == expected_ones, f"Ones failed: expected {expected_ones}, got {o.data}"

    i = identity(3)
    expected_identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert i.data == expected_identity, f"Identity failed: expected {expected_identity}, got {i.data}"

    print("✓ PASSED")


def test_non_square_multiply():
    """Test multiplication with non-square matrices"""
    print("Testing non-square matrix multiplication...", end=" ")

    # (2x3) @ (3x2) = (2x2)
    a = Matrix([[1, 2, 3],
                [4, 5, 6]])
    b = Matrix([[7, 8],
                [9, 10],
                [11, 12]])
    c = a.dot(b)

    assert c.shape() == (2, 2), f"Shape should be (2, 2), got {c.shape()}"

    # Manual calculation:
    # [[1*7+2*9+3*11, 1*8+2*10+3*12],     [[58, 64],
    #  [4*7+5*9+6*11, 4*8+5*10+6*12]]  =   [139, 154]]
    expected = [[58, 64], [139, 154]]
    assert c.data == expected, f"Expected {expected}, got {c.data}"

    print("✓ PASSED")


def test_identity_property():
    """Test that A @ I = A"""
    print("Testing identity property (A @ I = A)...", end=" ")

    a = Matrix([[1, 2], [3, 4]])
    i = identity(2)
    b = a.dot(i)

    assert b.data == a.data, f"A @ I should equal A. Expected {a.data}, got {b.data}"

    print("✓ PASSED")


def bonus_test_commutative():
    """Bonus test: Matrix multiplication is NOT commutative"""
    print("\nBonus: Testing non-commutativity (A @ B ≠ B @ A)...", end=" ")

    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[5, 6], [7, 8]])

    ab = a.dot(b)
    ba = b.dot(a)

    # These should be different!
    assert ab.data != ba.data, "A @ B should not equal B @ A (matrix multiply is not commutative)"

    print("✓ PASSED")
    print("  (A @ B =", ab.data, ")")
    print("  (B @ A =", ba.data, ")")
    print("  → Matrix multiplication is NOT commutative!")


def run_all_tests():
    """Run all test cases"""
    print("\n" + "="*60)
    print(" 🧪 TESTING MATRIX IMPLEMENTATION")
    print("="*60 + "\n")

    tests = [
        test_creation,
        test_addition,
        test_subtraction,
        test_element_wise_multiply,
        test_dot_product,
        test_transpose,
        test_apply,
        test_special_matrices,
        test_non_square_multiply,
        test_identity_property,
    ]

    failed = []

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ FAILED")
            print(f"  Error: {e}")
            failed.append(test.__name__)

    # Bonus test
    try:
        bonus_test_commutative()
    except Exception as e:
        print(f"✗ FAILED: {e}")

    print("\n" + "="*60)

    if not failed:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("\nExcellent work! You've successfully implemented matrix operations")
        print("from scratch. This is a HUGE milestone!")
        print("\nNext steps:")
        print("  1. Try multiplying 100x100 matrices and time it")
        print("  2. Show your code to your teacher (me!)")
        print("  3. Answer the understanding questions in WEEK1_EXERCISE.md")
        print("  4. Move on to Exercise 1.2: Activation Functions!")
    else:
        print(f"❌ {len(failed)} test(s) failed:")
        for name in failed:
            print(f"  - {name}")
        print("\nDon't worry! Debugging is part of learning.")
        print("Read the error messages carefully and try again.")

    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()
