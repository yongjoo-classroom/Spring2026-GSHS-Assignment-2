import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from logistic_regression import logistic_regression

def test_logistic_regression_1():
    '''
    Test Case: 1: Toy Dataset
    Predict whether a customer will buy (1) or not buy (0)
    given their age and income level
    '''
    # X = [age, income_level]
    X = np.array([ [18, 1], [22, 1], [25, 2], [28, 2], [30, 3],
      [35, 3], [40, 3], [45, 2], [50, 2], [55, 1] ])

    # y = buy (1) or not buy (0) 
    y = np.array([ 0, 0, 0, 1, 1, 1, 1, 1, 0, 0 ])

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Implement the logistic regression model
    y_pred = logistic_regression(x_train, y_train, x_test)

    # Assert accuracy greater than 65%
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.65, f"Test failed: Accuracy {accuracy} is less than 65%"

def test_logistic_regression_2():
    '''
    Test Case: 2: Iris Dataset
    Predict whether a flower is setosa (1) or not (0)
    given its petal length and petal width
    '''
    # Load the Iris dataset 
    iris = load_iris()
    x, y = iris.data, (iris.target == 0).astype(int) 

    # reduce the dimension of training sample to 2 (petal length + petal width)     
    x = x[:, [2, 3]].reshape(-1, 2)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    # Implement the logistic regression model
    y_pred = logistic_regression(x_train, y_train, x_test)

    # Assert accuracy greater than 99%
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.99, f"Test failed: Accuracy {accuracy} is less than 99%"

