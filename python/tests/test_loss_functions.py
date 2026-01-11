"""Test loss functions"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.loss import (
    mean_squared_error,
    mse_derivative,
    binary_cross_entropy,
    categorical_cross_entropy,
    one_hot_encode
)


def test_mse():
    print("Testing MSE...", end=" ")

    # Perfect prediction
    loss = mean_squared_error([1, 2, 3], [1, 2, 3])
    assert abs(loss) < 1e-6, "Perfect prediction should have 0 loss"

    # Known case
    loss = mean_squared_error([0, 0], [1, 1])
    assert abs(loss - 1.0) < 1e-6, "MSE([0,0], [1,1]) should be 1.0"

    print("✓ PASSED")


def test_binary_cross_entropy():
    print("Testing Binary Cross-Entropy...", end=" ")

    # Good prediction (true=1, pred=0.9)
    loss = binary_cross_entropy(1, 0.9)
    assert loss < 0.2, "Good prediction should have low loss"

    # Bad prediction (true=1, pred=0.1)
    loss = binary_cross_entropy(1, 0.1)
    assert loss > 2.0, "Bad prediction should have high loss"

    print("✓ PASSED")


def test_categorical_cross_entropy():
    print("Testing Categorical Cross-Entropy...", end=" ")

    # True label: class 1
    y_true = [0, 1, 0]

    # Good prediction
    y_pred = [0.1, 0.8, 0.1]
    loss = categorical_cross_entropy(y_true, y_pred)
    assert loss < 0.3, "Good prediction should have low loss"

    # Bad prediction
    y_pred = [0.4, 0.1, 0.5]
    loss = categorical_cross_entropy(y_true, y_pred)
    assert loss > 2.0, "Bad prediction should have high loss"

    print("✓ PASSED")


def test_one_hot_encode():
    print("Testing one-hot encoding...", end=" ")

    assert one_hot_encode(0, 3) == [1, 0, 0]
    assert one_hot_encode(2, 5) == [0, 0, 1, 0, 0]
    assert sum(one_hot_encode(3, 10)) == 1

    print("✓ PASSED")


def run_all_tests():
    print("\n" + "="*60)
    print(" 🧪 TESTING LOSS FUNCTIONS")
    print("="*60 + "\n")

    tests = [
        test_mse,
        test_binary_cross_entropy,
        test_categorical_cross_entropy,
        test_one_hot_encode,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("🎉 ALL LOSS TESTS PASSED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()