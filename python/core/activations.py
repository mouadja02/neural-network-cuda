"""
Activation functions and their derivatives.

Each function works on:
- Scalars (single numbers)
- Lists (element-wise)
- Matrix objects (element-wise)
"""

import math

from core.matrix import Matrix


def sigmoid(x):
    """
    Sigmoid activation: maps input to (0, 1)

    Formula: σ(x) = 1 / (1 + e^(-x))

    Properties:
    - Output range: (0, 1)
    - Smooth, differentiable
    - Problem: vanishing gradients for |x| > 5

    Use: Binary classification, gates in LSTMs

    Args:
        x: number, list, or Matrix

    Returns:
        Same type as input, with sigmoid applied element-wise

    Examples:
        sigmoid(0) = 0.5
        sigmoid(large positive) ≈ 1
        sigmoid(large negative) ≈ 0
    """
    # 1. Check if x is a number, list, or Matrix
    if isinstance(x, (int, float)):
        return 1 / (1 + math.exp(-x))
    elif isinstance(x, list):
        return [sigmoid(xi) for xi in x]
    elif isinstance(x, Matrix):
        return x.apply(sigmoid)
    else:
        raise TypeError("Input must be a number, list, or Matrix.")


def sigmoid_derivative(x):
    """
    Derivative of sigmoid.

    Formula: σ'(x) = σ(x) * (1 - σ(x))

    This is why sigmoid is convenient - its derivative is simple!

    Args:
        x: number, list, or Matrix (can be pre-computed sigmoid values)

    Returns:
        Derivative at each point

    Note: This function expects the OUTPUT of sigmoid (not the input).
    So you'd call it like: sigmoid_derivative(sigmoid(x))
    """
    if isinstance(x, (int, float)):
        return x * (1 - x)
    elif isinstance(x, list):
        return [sigmoid_derivative(xi) for xi in x]
    elif isinstance(x, Matrix):
        return x.apply(sigmoid_derivative)
    else:
        raise TypeError("Input must be a number, list, or Matrix.")


def relu(x):
    """
    ReLU (Rectified Linear Unit): most popular activation

    Formula: f(x) = max(0, x)

    Properties:
    - Output range: [0, ∞)
    - Dead simple, very fast
    - No vanishing gradient for x > 0
    - Problem: "dead neurons" (always output 0)

    Use: Hidden layers in most modern networks

    Examples:
        relu(-5) = 0
        relu(0) = 0
        relu(5) = 5
    """
    if isinstance(x, (int, float)):
        return max(0, x)
    elif isinstance(x, list):
        return [relu(xi) for xi in x]
    elif isinstance(x, Matrix):
        return x.apply(relu)
    else:
        raise TypeError("Input must be a number, list, or Matrix.")


def relu_derivative(x):
    """
    Derivative of ReLU.

    Formula: f'(x) = 1 if x > 0 else 0

    Technically undefined at x=0, but we use 0 by convention.

    Note: This expects the INPUT to ReLU (not the output).
    """
    if isinstance(x, (int,float)):
        return 1 if x > 0 else 0
    elif isinstance(x, list):
        return [relu_derivative(xi) for xi in x]
    elif isinstance(x, Matrix):
        return x.apply(relu_derivative)
    else:
        raise TypeError("Input must be a number, list, or Matrix.")


def tanh(x):
    """
    Hyperbolic tangent: zero-centered sigmoid

    Formula: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

    Properties:
    - Output range: (-1, 1)
    - Zero-centered (better than sigmoid)
    - Still suffers from vanishing gradients

    Use: RNNs, when you want zero-centered output

    Python has math.tanh() but implement it yourself!
    """
    if isinstance(x, (int, float)):
        e_pos, e_neg = math.exp(x), math.exp(-x)
        return (e_pos - e_neg) / (e_pos + e_neg)
    elif isinstance(x, list):
        return [tanh(xi) for xi in x]
    elif isinstance(x, Matrix):
        return x.apply(tanh)
    else:
        raise TypeError("Input must be a number, list, or Matrix.")


def tanh_derivative(x):
    """
    Derivative of tanh.

    Formula: tanh'(x) = 1 - tanh²(x)

    Note: Expects the OUTPUT of tanh.
    """
    if isinstance(x, (int, float)):
        return 1 - x ** 2
    elif isinstance(x, list):
        return [tanh_derivative(xi) for xi in x]
    elif isinstance(x, Matrix):
        return x.apply(tanh_derivative)
    else:
        raise TypeError("Input must be a number, list, or Matrix.")

def softmax(x):
    """
    Softmax: converts logits to probability distribution

    Formula: softmax(x_i) = e^(x_i) / Σ(e^(x_j))

    Properties:
    - Outputs sum to 1
    - Each output in (0, 1)
    - Differentiable

    Use: Multi-class classification (output layer)

    IMPORTANT: Numerical stability!
    e^x explodes for large x. Use this trick:
    softmax(x) = softmax(x - max(x))

    This doesn't change the result but prevents overflow.

    Args:
        x: list or Matrix (treats as vector or row vectors)

    Returns:
        Same shape, normalized to probability distribution

    Examples:
        softmax([1, 2, 3]) ≈ [0.09, 0.24, 0.67]
        Notice: sum = 1.0
    """
    if isinstance(x, list):
        x_shifted = [xi - max(x) for xi in x]
        exp_x = [math.exp(xi) for xi in x_shifted]
        sum_exp_x = sum(exp_x)
        return [xi / sum_exp_x for xi in exp_x]
    elif isinstance(x, Matrix):
        result = []
        for i in range(x.rows):
            row = x.data[i]
            row_softmax = softmax(row)
            result.append(row_softmax)
        return Matrix(result)
    else:
        raise TypeError("Input must be a list or Matrix.")


def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU: fixes "dead neurons" problem

    Formula: f(x) = x if x > 0 else alpha * x

    Properties:
    - Allows small gradient when x < 0
    - No dead neurons
    - alpha typically 0.01

    Use: Alternative to ReLU in hidden layers
    """
    if isinstance(x, (int, float)):
        return x if x > 0 else alpha * x
    elif isinstance(x, list):
        return [leaky_relu(xi) for xi in x]
    elif isinstance(x, Matrix):
        return x.apply(leaky_relu)
    else:
        raise TypeError("Input must be a number, list, or Matrix.")




def leaky_relu_derivative(x, alpha=0.01):
    """
    Derivative of Leaky ReLU.

    Formula: f'(x) = 1 if x > 0 else alpha
    """
    if isinstance(x, (int,float)):
        return 1 if x > 0 else alpha
    elif isinstance(x, list):
        return [leaky_relu_derivative(xi) for xi in x]
    elif isinstance(x, Matrix):
        return x.apply(leaky_relu_derivative)
    else:
        raise TypeError("Input must be a number, list, or Matrix.")



# Helper function for type checking
def _handle_types(func, x):
    """
    Helper to apply activation function to different types.

    You can use this in your implementations!
    """
    from core.matrix import Matrix

    if isinstance(x, Matrix):
        # Use the apply() method from your Matrix class!
        return x.apply(func)
    elif isinstance(x, list):
        # Recursively handle nested lists
        return [_handle_types(func, xi) for xi in x]
    else:
        # Assume it's a number
        return func(x)