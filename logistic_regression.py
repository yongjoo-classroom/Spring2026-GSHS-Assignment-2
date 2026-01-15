import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Logistic:
    def __init__(self, lr):
        self.lr = lr
        self.w = np.zeros(2)
        self.b = 0

    def forward(self, x):
        return sigmoid(np.dot(x, self.w) + self.b)

    def predict(self, x):
        return (self.forward(x) >= 0.5).astype(int)

    def train(self, x_t, y_t, epoch):
        n_samples = len(y_t)
        for _ in range(epoch):
            y_hat = self.forward(x_t)
            grad_w = np.dot(x_t.T, y_hat - y_t) / n_samples
            grad_b = np.sum(y_hat - y_t) / n_samples
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b


def logistic_regression(x_train,y_train,x_test):
    model = Logistic(0.05)
    model.train(x_train, y_train, 10000)
    return model.predict(x_test)