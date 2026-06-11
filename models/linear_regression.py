import numpy as np

class LinearRegression:
    def __init__(self, learning_rate = 0.01, n_iters = 1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        
    def fit(self, X, y): # Currently only supports BGD, later add SGD and MBGD
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))
        self.weights = np.zeros((X.shape[1]))
        for _ in range(self.n_iters):
            predictions = X @ self.weights
            errors = predictions - y
            SumSquaredError = np.sum(errors ** 2)
            gradient = (X.T @ errors) / X.shape[0] # Normalize using m, different from CS229 notes
            self.weights = self.weights - self.learning_rate * gradient
            if _ % 1000 == 0:
                print(SumSquaredError)
        return self
    
    def predict(self, X):
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))
        return X @ self.weights

        









