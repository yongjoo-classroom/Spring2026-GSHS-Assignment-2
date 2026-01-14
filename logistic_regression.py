import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    n_samples, n_features = x_train.shape
    w = np.zeros(n_features)
    b = 0.0

    lr = 0.1
    epochs = 10000

    for epoch in range(epochs):
        linear = np.dot(x_train, w) + b
        y_hat = sigmoid(linear)

        dw = (1 / n_samples) * np.dot(x_train.T, (y_hat - y_train))
        db = (1 / n_samples) * np.sum(y_hat - y_train)

        w -= lr * dw
        b -= lr * db

        lr *= 0.999

    linear_test = np.dot(x_test, w) + b
    y_prob = sigmoid(linear_test)

    y_pred = (y_prob >= 0.5).astype(int)
    return y_pred