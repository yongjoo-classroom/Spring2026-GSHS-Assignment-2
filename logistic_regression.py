import numpy as np

def logistic_regression(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    '''
    Implements the logistic regression algorithm.

    Parameters:
        - x_train: Training features of shape (n_samples, 2).
                    For this assignment, each training sample has two features: [feature1, feature2]
        - y_train: Training labels (0/1) 
                    All the predictions will be binary (0 or 1), since it is Logistic Regression.
        - x_test: Test features of shape (n_samples, 2).

    Returns:
        y_pred: Predicted labels for the test set
    '''
    # Your code here
    lr = 0.1 
    epochs = 1000
    n,d = x_train.shape
    X = np.hstack([np.ones((n, 1)), x_train])
    X_test = np.hstack([np.ones((x_test.shape[0], 1)), x_test])
    w = np.zeros(d + 1)
    def sigmoid(z): return 1 / (1 + np.exp(-z))
    for _ in range(epochs):
        z = X @ w
        y_pred = sigmoid(z)
        grad = (1 / n) * X.T @ (y_pred - y_train)
        w -= lr * grad
    probs = sigmoid(X_test @ w)
    y_pred = (probs >= 0.5).astype(int)
    return y_pred
