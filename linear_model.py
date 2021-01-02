import numpy as np
from function import sigmoid


class LogisticRegression(object):
    def __init__(self):
        self._coef = None
        self._intercept = None

    def fit(self, x, y, learning_rate=0.01, epochs=1000, lim=1):
        x = np.insert(x, 0, values=1, axis=1)
        self._weights_initializer()
        weights = np.concatenate([self._intercept, self._coef])
        for _ in range(epochs):
            h = sigmoid(np.dot(x, weights.reshape((-1, 1))))
            error = h - y.reshape((-1, 1))
            weights = weights - learning_rate * np.dot(np.transpose(x), error).reshape(-1)
            self._coef = weights[1:]
            self._intercept = np.array([weights[0]])
            if self._loss(x, y) < lim:
                break
        print("loss: ", self._loss(x, y))

    def _loss(self, x, y):
        loss = 0
        weights = np.concatenate([self._intercept, self._coef])
        for i in range(len(x)):
            theta = weights.reshape(1, 3).dot(x[i])[0]
            prob = sigmoid(theta)
            loss += -y[i] * np.log(prob) - (1 - y[i]) * np.log(1 - prob)
        return loss

    def _weights_initializer(self, method="one"):
        if method == "one":
            self._coef = np.ones(2)
            self._intercept = np.ones(1)
