# Logistic Regression for data classification

In this assignment, you will implement the Logistic Regression algorithm to classify given dataset by learning a decision boundary that separates data points into different classes based on their feature values.

## Assignment Instructions

### Objective
Implement the `logistic_regression` function in `logistic_regression.py` to learn a classification model from the training data and use it to predict the labels of the test set. The function should output:
1. Predicted labels of the `x_test` dataset.

### Input Format
- `x_train`: A `numpy.ndarray` of shape `(num_training_values, 2)` where each row represents two features of a given data point.
- `y_train`: A `numpy.ndarray` of shape `(num_training_values)` where `y_train[i]` is the expected label for `x_train[i]` data point.
- `x_test`: A `numpy.ndarray` of shape `(num_test_values, 2)` where each row represents two features of a given data point. 
            The labels are to be predicted for these datapoints.

### Output Format
- Predicted labels for `x_test` as a `numpy.ndarray` of shape `len(x_test)`.

### Starter Code
The starter code is in `logistic_regression.py`. You can import libraries like sklearn for implementing logistic regression.

## Testing
Given the predicted labels for `x_test`, the model is evaluated using accuracy, computed as:
```
Accuracy = (Number of correctly predicted labels) / (Total number of test data points)
```
All test cases are considered passed if the accuracy exceeds a predefined threshold.

**Dataset Used:** 
- Iris Dataset
- Toy Dataset(sample)
