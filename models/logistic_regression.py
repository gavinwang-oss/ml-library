import numpy as np

class LogisticRegression():
    def __init__(self, learning_rate = 0.01, n_iters = 1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
    

    def fit(self, X, y):
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))
        self.weights = np.zeros(X.shape[1])
        for _ in range(self.n_iters):
            z = X @ self.weights # z is m x 1
            hypotheses = 1 / (1 + np.exp(-z))
            self.weights = self.weights + self.learning_rate * (1 / X.shape[0]) * (X.T @ (y - hypotheses))
        return self

    def predict(self, X, threshold = 0.5):
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))
        hypotheses = 1 / (1 + np.exp(-(X @ self.weights)))
        predictions = (hypotheses >= threshold).astype(int)
        return predictions
    






