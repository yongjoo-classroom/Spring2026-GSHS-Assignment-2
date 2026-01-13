import numpy as np

def logistic_regression(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    '''
    Implements the logistic regression algorithm.

    Parameters:
        - x_train: Training features of shape (n_samples, 2).
        - y_train: Training labels (0/1)
        - x_test: Test features of shape (n_samples, 2).

    Returns:
        y_pred: Predicted labels for the test set as a numpy.ndarray of shape (len(x_test),)
    '''
    # Lazy imports so the file still imports even if sklearn isn't loaded until runtime
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    # Ensure correct shapes/dtypes
    x_train = np.asarray(x_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    y_train = np.asarray(y_train).ravel().astype(int)

    # Create and train model
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, solver="lbfgs")
    )
    model.fit(x_train, y_train)

    # Predict labels (0/1)
    y_pred = model.predict(x_test).astype(int)
    return y_pred
