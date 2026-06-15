import numpy as np

class Perceptron():
    def __init__(self, learning_rate = 0.01, n_iters = 1000):
        self.learning_rate = learning_rate
        self.weights = None
        self.n_iters = n_iters

    def fit(self, X, y):
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))
        self.weights = np.zeros(X.shape[1])
        for _ in range(self.n_iters):
            for row in range(X.shape[0]):
                prediction = 1 if X[row] @ self.weights.T >= 0 else 0
                self.weights = self.weights + self.learning_rate * (y[row] - prediction) * X[row]
        return self
    
    def predict(self, X):
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))
        hx = X @ self.weights.T 
        hx = (hx >= 0).astype(int)
        return hx












