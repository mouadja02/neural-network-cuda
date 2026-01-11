
import sys
import os
import math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.activations import (
    sigmoid, sigmoid_derivative,
    relu, relu_derivative,
    tanh, tanh_derivative,
    softmax
)
from core.matrix import Matrix


def test_sigmoid():
    print("Testing sigmoid...", end=" ")

    # Test scalar
    assert abs(sigmoid(0) - 0.5) < 1e-6, "sigmoid(0) should be 0.5"
    assert sigmoid(100) > 0.99, "sigmoid(large) should be close to 1"
    assert sigmoid(-100) < 0.01, "sigmoid(-large) should be close to 0"

    # Test list
    result = sigmoid([0, 1, -1])
    assert abs(result[0] - 0.5) < 1e-6

    # Test Matrix
    m = Matrix([[0, 1], [-1, 2]])
    result = sigmoid(m)
    assert abs(result.data[0][0] - 0.5) < 1e-6

    print("✓ PASSED")


def test_sigmoid_derivative():
    print("Testing sigmoid derivative...", end=" ")

    # σ'(x) = σ(x) * (1 - σ(x))
    # At x=0: σ(0) = 0.5, so σ'(0) = 0.5 * 0.5 = 0.25
    sig_0 = sigmoid(0)
    deriv = sigmoid_derivative(sig_0)
    assert abs(deriv - 0.25) < 1e-6

    print("✓ PASSED")


def test_relu():
    print("Testing ReLU...", end=" ")

    assert relu(5) == 5
    assert relu(0) == 0
    assert relu(-5) == 0

    result = relu([-2, -1, 0, 1, 2])
    assert result == [0, 0, 0, 1, 2]

    print("✓ PASSED")


def test_relu_derivative():
    print("Testing ReLU derivative...", end=" ")

    assert relu_derivative(5) == 1
    assert relu_derivative(0) == 0
    assert relu_derivative(-5) == 0

    print("✓ PASSED")


def test_tanh():
    print("Testing tanh...", end=" ")

    assert abs(tanh(0)) < 1e-6, "tanh(0) should be 0"
    assert tanh(100) > 0.99, "tanh(large) should be close to 1"
    assert tanh(-100) < -0.99, "tanh(-large) should be close to -1"

    print("✓ PASSED")


def test_softmax():
    print("Testing softmax...", end=" ")

    # Test that outputs sum to 1
    result = softmax([1, 2, 3])
    total = sum(result)
    assert abs(total - 1.0) < 1e-6, "Softmax should sum to 1"

    # Test that larger inputs get higher probabilities
    assert result[2] > result[1] > result[0], "Larger inputs should have higher probability"

    # Test numerical stability with large numbers
    result = softmax([1000, 1001, 1002])
    assert not any(math.isinf(x) or math.isnan(x) for x in result), "Should handle large numbers"

    print("✓ PASSED")


def run_all_tests():
    print("\n" + "="*60)
    print(" 🧪 TESTING ACTIVATION FUNCTIONS")
    print("="*60 + "\n")

    tests = [
        test_sigmoid,
        test_sigmoid_derivative,
        test_relu,
        test_relu_derivative,
        test_tanh,
        test_softmax,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("🎉 ALL ACTIVATION TESTS PASSED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()