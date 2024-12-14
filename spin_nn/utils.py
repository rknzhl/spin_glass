import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
def load_mnist():
    mnist = fetch_openml("mnist_784", version=1)
    X = mnist.data / 255.0  # Normalize data
    y = mnist.target.astype(np.int32)
    return train_test_split(X, y, test_size=0.2, random_state=42)