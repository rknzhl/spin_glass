import numpy as np
import pandas as pd
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

    X[X == 0] = -1
    y[y == 0] = -1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.to_numpy()
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.to_numpy()

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    return X_train, X_test, y_train, y_test