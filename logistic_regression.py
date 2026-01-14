import numpy as np
import math

def logistic_regression(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    lr = 0.1
    epochs = 5000

    # numpy array로 변환 (안전)
    X = np.array(x_train, dtype=float)
    y = np.array(y_train, dtype=float)
    X_test = np.array(x_test, dtype=float)

    # ===== feature normalization (중요) =====
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X = (X - mean) / std
    X_test = (X_test - mean) / std

    n_samples, n_features = X.shape

    # weight + bias
    w = np.zeros(n_features)
    b = 0.0

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    # Gradient Descent
    for _ in range(epochs):
        z = np.dot(X, w) + b
        y_pred = sigmoid(z)

        dw = np.dot(X.T, (y_pred - y)) / n_samples
        db = np.sum(y_pred - y) / n_samples

        w -= lr * dw
        b -= lr * db

    # Test prediction
    z_test = np.dot(X_test, w) + b
    y_test_pred = sigmoid(z_test)

    return (y_test_pred >= 0.5).astype(int)
