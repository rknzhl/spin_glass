import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def load_mnist():
    mnist = fetch_openml("mnist_784", version=1)
    X = mnist.data / 255.0 
    y = mnist.target.astype(np.int32).to_numpy()
    # One-Hot Encoding меток
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y.reshape(-1, 1))
    return train_test_split(X, y, test_size=0.2, random_state=42)