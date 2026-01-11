"""
Matrix operations from scratch - No NumPy!

This is your implementation. Fill in the TODO sections.
"""


class Matrix:
    """
    A simple matrix implementation using nested Python lists.
    """

    def __init__(self, data):
        """Initialize matrix from 2D list."""
        # TODO: Your code here
        # Hints:
        # 1. Validate that all rows have the same length
        # 2. Store the data
        # 3. Calculate and store rows and cols
        pass

    def shape(self):
        """Return tuple (rows, cols)"""
        # TODO: Your code here
        pass

    def __add__(self, other):
        """Add two matrices element-wise."""
        # TODO: Your code here
        pass

    def __sub__(self, other):
        """Subtract two matrices element-wise."""
        # TODO: Your code here
        pass

    def __mul__(self, other):
        """Element-wise multiplication (Hadamard product)."""
        # TODO: Your code here
        pass

    def dot(self, other):
        """Matrix multiplication."""
        # TODO: Your code here
        # This is the hardest one - take your time!
        pass

    def T(self):
        """Transpose matrix."""
        # TODO: Your code here
        pass

    def apply(self, func):
        """Apply function to every element."""
        # TODO: Your code here
        pass

    def __repr__(self):
        """String representation."""
        # TODO: Make it print nicely
        return f"Matrix({self.data})"

    def __eq__(self, other):
        """Check equality."""
        # TODO: Your code here
        pass


def zeros(rows, cols):
    """Create matrix of zeros."""
    # TODO: Your code here
    pass


def ones(rows, cols):
    """Create matrix of ones."""
    # TODO: Your code here
    pass


def identity(n):
    """Create identity matrix."""
    # TODO: Your code here
    pass
