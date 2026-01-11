"""
Loss functions and their derivatives.

These measure how wrong our predictions are.
"""

import math

from core.matrix import Matrix

def mean_squared_error(y_true, y_pred):
    """
    Mean Squared Error: for regression tasks

    Formula: MSE = (1/n) * Σ(y_true - y_pred)²

    Properties:
    - Always non-negative
    - Penalizes large errors heavily (squared)
    - Differentiable everywhere

    Use: Regression (predicting continuous values)

    Args:
        y_true: list or Matrix of true values
        y_pred: list or Matrix of predicted values

    Returns:
        Single number (the loss)

    Examples:
        mse([1,2,3], [1,2,3]) = 0.0  (perfect)
        mse([0,0], [1,1]) = 1.0
    """
    if isinstance(y_true, Matrix) and isinstance(y_pred, Matrix):
        if y_true.shape() != y_pred.shape():
            raise ValueError("y_pred and y_true must have the same dimensions to calculate the MSE.")
        sum = 0
        cols, rows = y_pred.cols, y_pred.rows
        for i in range(rows):
            for j in range(cols):
                sum += (y_pred[i][j] - y_true[i][j]) ** 2 
        return (1/(rows * cols)) * sum 
    elif isinstance(y_true, list) and isinstance(y_pred, list):
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred dimensions don't match !")
        sum = 0
        n = len(y_true)
        for i in range(n):
            sum += (y_pred[i] - y_true[i]) ** 2 
        return sum / n
    else:
        raise TypeError("Input must be a Matrix or a list")


def mse_derivative(y_true, y_pred):
    """
    Derivative of MSE with respect to predictions.

    Formula: dMSE/dy_pred = (2/n) * (y_pred - y_true)

    This tells us which direction to adjust predictions.

    Returns:
        Same shape as inputs
    """
    if isinstance(y_true, Matrix) and isinstance(y_pred, Matrix):
        if y_true.shape() != y_pred.shape():
            raise ValueError("y_pred and y_true must have the same dimensions to calculate the MSE.")
        result = 0
        cols, rows = y_pred.cols, y_pred.rows
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append((2/(rows * cols)) * (y_pred[i][j] - y_true[i][j])) 
            result.append(row)
        return Matrix(result)
    elif isinstance(y_true, list) and isinstance(y_pred, list):
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred dimensions don't match !")
        M_y_true, M_y_pred = Matrix(y_true), Matrix(y_pred)
        return mse_derivative(M_y_true, M_y_pred)
    else:
        raise TypeError("Input must be a Matrix or a list")


def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    Binary Cross-Entropy: for binary classification

    Formula: BCE = -[y*log(ŷ) + (1-y)*log(1-ŷ)]

    Properties:
    - Heavily penalizes confident wrong predictions
    - Works with sigmoid output

    Use: Binary classification (yes/no, spam/not spam)

    Args:
        y_true: 0 or 1 (or list/Matrix of 0s and 1s)
        y_pred: probability in (0, 1)
        epsilon: small constant for numerical stability

    IMPORTANT: Add epsilon to prevent log(0) = -∞
    Use: log(y_pred + epsilon) and log(1 - y_pred + epsilon)

    Examples:
        bce(1, 0.9) ≈ 0.105  (good prediction, low loss)
        bce(1, 0.1) ≈ 2.303  (bad prediction, high loss)
    """
    # TODO: Implement
    pass


def binary_cross_entropy_derivative(y_true, y_pred, epsilon=1e-15):
    """
    Derivative of BCE.

    Formula: dBCE/dŷ = (ŷ - y) / (ŷ * (1 - ŷ))

    When combined with sigmoid, this simplifies nicely!
    """
    # TODO: Implement
    pass


def categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    Categorical Cross-Entropy: for multi-class classification

    Formula: CCE = -Σ(y_true * log(y_pred))

    Properties:
    - Works with one-hot encoded labels
    - Works with softmax output
    - Output is a probability distribution

    Use: Multi-class classification (digit recognition!)

    Args:
        y_true: one-hot vector, e.g., [0, 0, 1, 0] for class 2
        y_pred: probability distribution, e.g., [0.1, 0.2, 0.6, 0.1]
        epsilon: prevents log(0)

    Examples:
        True label: [0, 1, 0]  (class 1)
        Prediction: [0.1, 0.8, 0.1]  → loss ≈ 0.22  (good)
        Prediction: [0.4, 0.1, 0.5]  → loss ≈ 2.30  (bad)
    """
    # TODO: Implement
    # Hint: Only non-zero y_true values contribute to the sum
    pass


def categorical_cross_entropy_derivative(y_true, y_pred, epsilon=1e-15):
    """
    Derivative of categorical cross-entropy.

    Formula: dCCE/dŷ = -y_true / y_pred

    When combined with softmax, simplifies to: ŷ - y_true
    """
    # TODO: Implement
    pass


# Utility function for later
def one_hot_encode(label, num_classes):
    """
    Convert integer label to one-hot vector.

    Args:
        label: integer in [0, num_classes)
        num_classes: total number of classes

    Returns:
        List with 1 at position 'label', 0 elsewhere

    Examples:
        one_hot_encode(2, 5) → [0, 0, 1, 0, 0]
        one_hot_encode(0, 3) → [1, 0, 0]
    """
    # TODO: Implement
    pass