import math

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
    epochs = 5000

    n_samples = len(x_train)
    n_features = len(x_train[0])

    # weight 초기화
    w = [0.0] * n_features

    def sigmoid(z):
        return 1 / (1 + math.exp(-z))

    # Gradient Descent
    for _ in range(epochs):
        dw = [0.0] * n_features

        for i in range(n_samples):
            z = 0.0
            for j in range(n_features):
                z += x_train[i][j] * w[j]

            y_pred = sigmoid(z)
            error = y_pred - y_train[i]

            for j in range(n_features):
                dw[j] += error * x_train[i][j]

        for j in range(n_features):
            w[j] -= lr * dw[j] / n_samples

    # Test prediction
    y_pred = []
    for x in x_test:
        z = 0.0
        for j in range(n_features):
            z += x[j] * w[j]
        y_pred.append(1 if sigmoid(z) >= 0.5 else 0)

    return y_pred
