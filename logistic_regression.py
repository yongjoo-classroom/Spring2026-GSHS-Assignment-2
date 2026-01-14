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
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    x_test = np.asarray(x_test, dtype=float)

    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    sigma[sigma == 0] = 1.0

    x_train_s = (x_train - mu) / sigma
    x_test_s = (x_test - mu) / sigma

    n = x_train_s.shape[0]
    X = np.hstack([np.ones((n, 1)), x_train_s])
    Xt = np.hstack([np.ones((x_test_s.shape[0], 1)), x_test_s])

    w = np.zeros(X.shape[1], dtype=float)

    lr = 0.2
    iterations = 20000
    l2 = 0.01

    for _ in range(iterations):
        z = X @ w
        z = np.clip(z, -50, 50)
        p = 1.0 / (1.0 + np.exp(-z))

        grad = (X.T @ (p - y_train)) / n
        reg = l2 * np.r_[0.0, w[1:]] / n
        w -= lr * (grad + reg)

    zt = Xt @ w
    zt = np.clip(zt, -50, 50)
    pt = 1.0 / (1.0 + np.exp(-zt))
    return (pt >= 0.5).astype(int)
