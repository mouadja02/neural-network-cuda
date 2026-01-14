"""
MNIST Digit Classification using GPU-Accelerated Neural Network
Full MNIST Dataset (60,000 training + 10,000 test images)

This script downloads the official MNIST dataset and trains using CUDA kernels.
"""

import os
import numpy as np
import gzip
import time
import pycuda.gpuarray as gpuarray
import matplotlib.pyplot as plt

from NeuralNetwork import NeuralNetworkGPU
from utils.progress import progress_bar


# ============================================================================
# MNIST DATA LOADING FUNCTIONS
# ============================================================================

def download_mnist_dataset():
    """
    Download MNIST dataset using kagglehub
    
    Returns:
        path: Path to downloaded dataset
    """
    print("\n" + "=" * 70)
    print(" Downloading MNIST Dataset")
    print("=" * 70)
    
    try:
        import kagglehub
        
        # Download latest version
        path = kagglehub.dataset_download("hojjatk/mnist-dataset")
        
        print(f"\n✓ Dataset downloaded successfully!")
        print(f"  Path: {path}")
        
        return path
    except ImportError:
        print("Install KaggleHub: pip install kagglehub")
        return None
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None


def load_mnist_images(filename):
    """
    Load MNIST images from idx3-ubyte format (gzipped or uncompressed)
    
    Args:
        filename: Path to file containing images
    
    Returns:
        images: NumPy array of shape (n_samples, 784)
    """
    print(f"\nLoading images from {os.path.basename(filename)}...")
    
    # Determine if file is gzipped
    is_gzipped = filename.endswith('.gz')
    
    if is_gzipped:
        open_func = gzip.open
    else:
        open_func = open
    
    with open_func(filename, 'rb') as f:
        # Read magic number and dimensions
        magic = int.from_bytes(f.read(4), 'big')
        n_images = int.from_bytes(f.read(4), 'big')
        n_rows = int.from_bytes(f.read(4), 'big')
        n_cols = int.from_bytes(f.read(4), 'big')
        
        print(f"  Magic number: {magic}")
        print(f"  Number of images: {n_images:,}")
        print(f"  Image size: {n_rows}×{n_cols}")
        
        # Read image data
        buf = f.read(n_images * n_rows * n_cols)
        images = np.frombuffer(buf, dtype=np.uint8)
        images = images.reshape(n_images, n_rows * n_cols)
        
        # Normalize to [0, 1]
        images = images.astype(np.float32) / 255.0
        
        print(f"  ✓ Loaded shape: {images.shape}")
        
        return images


def load_mnist_labels(filename):
    """
    Load MNIST labels from idx1-ubyte format (gzipped or uncompressed)
    
    Args:
        filename: Path to file containing labels
    
    Returns:
        labels: NumPy array of shape (n_samples,)
    """
    print(f"\nLoading labels from {os.path.basename(filename)}...")
    
    # Determine if file is gzipped
    is_gzipped = filename.endswith('.gz')
    
    if is_gzipped:
        open_func = gzip.open
    else:
        open_func = open
    
    with open_func(filename, 'rb') as f:
        # Read magic number and count
        magic = int.from_bytes(f.read(4), 'big')
        n_labels = int.from_bytes(f.read(4), 'big')
        
        print(f"  Magic number: {magic}")
        print(f"  Number of labels: {n_labels:,}")
        
        # Read label data
        buf = f.read(n_labels)
        labels = np.frombuffer(buf, dtype=np.uint8)
        
        print(f"  ✓ Loaded shape: {labels.shape}")
        print(f"  Label distribution: {np.bincount(labels)}")
        
        return labels


def load_full_mnist_dataset(dataset_path):
    """
    Load complete MNIST dataset
    
    Args:
        dataset_path: Path to directory containing MNIST files
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    print("\n" + "=" * 70)
    print(" Loading Full MNIST Dataset")
    print("=" * 70)
    
    print(f"\nSearching for MNIST files in: {dataset_path}")
    
    # Search for files recursively
    def find_file(base_path, pattern):
        """Recursively search for a file matching pattern"""
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if pattern in file:
                    return os.path.join(root, file)
        return None
    
    # Find all required files
    train_images_path = find_file(dataset_path, 'train-images')
    train_labels_path = find_file(dataset_path, 'train-labels')
    test_images_path = find_file(dataset_path, 't10k-images')
    test_labels_path = find_file(dataset_path, 't10k-labels')
    
    print(f"\n✓ Found all required files:")
    print(f"   Train images: {train_images_path}")
    print(f"   Train labels: {train_labels_path}")
    print(f"   Test images:  {test_images_path}")
    print(f"   Test labels:  {test_labels_path}")
    
    # Load training data
    X_train = load_mnist_images(train_images_path)
    y_train = load_mnist_labels(train_labels_path)
    
    # Load test data
    X_test = load_mnist_images(test_images_path)
    y_test = load_mnist_labels(test_labels_path)
    
    print("\n" + "=" * 70)
    print(" Dataset Summary")
    print("=" * 70)
    print(f"\nTraining set:")
    print(f"  Images: {X_train.shape}")
    print(f"  Labels: {y_train.shape}")
    print(f"\nTest set:")
    print(f"  Images: {X_test.shape}")
    print(f"  Labels: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_mnist_model(X_train, y_train, X_test, y_test, 
                      hidden_size=128, 
                      batch_size=64, 
                      n_epochs=10, 
                      learning_rate=0.01):
    """
    Train MNIST classifier using GPU-accelerated neural network
    
    Args:
        X_train: Training images (n_samples, 784)
        y_train: Training labels (n_samples,)
        X_test: Test images (n_samples, 784)
        y_test: Test labels (n_samples,)
        hidden_size: Number of neurons in hidden layer
        batch_size: Training batch size
        n_epochs: Number of training epochs
        learning_rate: Learning rate for SGD
    
    Returns:
        nn: Trained neural network
        history: Dictionary with training history
    """
    
    input_size = 784  # 28x28 pixels
    output_size = 10  # 10 digits (0-9)
    
    print("\n" + "=" * 70)
    print(" MNIST Training with GPU Neural Network")
    print("=" * 70)
    print(f"\nNetwork Architecture:")
    print(f"  Input:  {input_size} (28×28 pixels)")
    print(f"  Hidden: {hidden_size} (ReLU activation)")
    print(f"  Output: {output_size} (Softmax activation)")
    print(f"\nTraining Configuration:")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Test samples:     {len(X_test):,}")
    print(f"  Batch size:       {batch_size}")
    print(f"  Epochs:           {n_epochs}")
    print(f"  Learning rate:    {learning_rate}")
    
    # Create network
    print("\nInitializing GPU neural network...")
    nn = NeuralNetworkGPU(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        batch_size=batch_size
    )
    
    # Initial evaluation
    print("\nEvaluating initial accuracy (random weights)...")
    initial_test_acc = nn.evaluate(X_test[:1000], y_test[:1000])
    print(f"  Initial test accuracy: {initial_test_acc*100:.2f}%")
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'epoch_time': []
    }
    
    # Training loop
    print("\n" + "=" * 70)
    print(" Training Started")
    print("=" * 70)
    
    n_batches = len(X_train) // batch_size
    
    for epoch in progress_bar(range(n_epochs)):
        epoch_start = time.time()
        epoch_loss = 0.0
        
        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        # Train on batches
        for batch_idx in range(n_batches):
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            # One-hot encode labels
            y_onehot = np.zeros((batch_size, output_size), dtype=np.float32)
            y_onehot[np.arange(batch_size), y_batch] = 1
            
            # Transfer to GPU
            d_X_batch = gpuarray.to_gpu(X_batch)
            d_y_batch = gpuarray.to_gpu(y_onehot)
            
            # Train on batch
            loss = nn.train_batch(d_X_batch, d_y_batch, learning_rate)
            epoch_loss += loss
            
            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs} | Batch {batch_idx+1}/{n_batches} | Loss: {loss:.4f}", end='\r')
        
        # Compute metrics
        avg_loss = epoch_loss / n_batches
        train_accuracy = nn.evaluate(X_train[:5000], y_train[:5000])  # Sample for speed
        test_accuracy = nn.evaluate(X_test, y_test)
        epoch_time = time.time() - epoch_start
        
        # Store history
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_accuracy)
        history['test_acc'].append(test_accuracy)
        history['epoch_time'].append(epoch_time)
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{n_epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_accuracy*100:5.2f}% | "
              f"Test Acc: {test_accuracy*100:5.2f}% | "
              f"Time: {epoch_time:5.2f}s")
    
    # Final evaluation
    print("\n" + "=" * 70)
    print(" Training Complete!")
    print("=" * 70)
    
    final_train_acc = nn.evaluate(X_train[:10000], y_train[:10000])
    final_test_acc = nn.evaluate(X_test, y_test)
    
    print(f"\nFinal Results:")
    print(f"  Train Accuracy: {final_train_acc*100:.2f}%")
    print(f"  Test Accuracy:  {final_test_acc*100:.2f}%")
    print(f"  Total Training Time: {sum(history['epoch_time']):.2f}s")
    print(f"  Average Time per Epoch: {np.mean(history['epoch_time']):.2f}s")
    
    return nn, history


def test_predictions(nn, X_test, y_test, n_samples=20):
    """
    Test predictions on random samples and display results
    
    Args:
        nn: Trained neural network
        X_test: Test images
        y_test: Test labels
        n_samples: Number of samples to test
    """
    print("\n" + "=" * 70)
    print(" Testing Random Predictions")
    print("=" * 70)
    
    # Select random samples
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    correct = 0
    for idx in indices:
        X_sample = X_test[idx:idx+1]
        y_true = y_test[idx]
        
        # Get prediction
        predictions = nn.predict(X_sample)
        y_pred = np.argmax(predictions[0])
        confidence = predictions[0][y_pred]
        
        # Display result
        status = "✓" if y_pred == y_true else "✗"
        if y_pred == y_true:
            correct += 1
        print(f"{status} True: {y_true} | Predicted: {y_pred} | Confidence: {confidence*100:.1f}%")
    
    print(f"\nSample Accuracy: {correct}/{n_samples} ({correct/n_samples*100:.1f}%)")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" MNIST Digit Classification - Full Dataset")
    print(" Using CUDA-Accelerated Neural Network")
    print("=" * 70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Download dataset
    dataset_path = download_mnist_dataset()
    
    if dataset_path is None:
        print("\n❌ Failed to download dataset. Exiting...")
        exit(1)
    
    # Load MNIST dataset
    X_train, y_train, X_test, y_test = load_full_mnist_dataset(dataset_path)
    
    # Normalize data
    print("\n" + "=" * 70)
    print(" Data Preprocessing")
    print("=" * 70)
    print("\nNormalizing data to zero mean and unit variance...")
    
    mean = X_train.mean()
    std = X_train.std()
    
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)
    
    print(f"  Training set - Mean: {X_train.mean():.6f}, Std: {X_train.std():.6f}")
    print(f"  Test set - Mean: {X_test.mean():.6f}, Std: {X_test.std():.6f}")
    
    # Train model
    nn, history = train_mnist_model(
        X_train, y_train,
        X_test, y_test,
        hidden_size=128,
        batch_size=64,
        n_epochs=20,
        learning_rate=0.01
    )
    
    # Test some predictions
    test_predictions(nn, X_test, y_test, n_samples=40)
    
    # Print summary
    print("\n" + "=" * 70)
    print(" Training Summary")
    print("=" * 70)
    print(f"\nBest Test Accuracy: {max(history['test_acc'])*100:.2f}% "
          f"(Epoch {np.argmax(history['test_acc'])+1})")
    print(f"Final Test Accuracy: {history['test_acc'][-1]*100:.2f}%")
    print(f"Improvement: {(history['test_acc'][-1] - history['test_acc'][0])*100:.2f}%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    ax1.plot(history['train_loss'])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['test_acc'], label='Test')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('mnist_training_curves.png', dpi=150)
        
    print("\n" + "=" * 70)
    print(" MNIST Training Complete! 🎉")
    print("=" * 70)
    print(f"\n🎯 Final Test Accuracy: {history['test_acc'][-1]*100:.2f}%")
    print(f"⚡ Total Training Time: {sum(history['epoch_time']):.1f}s")
    print("=" * 70)
