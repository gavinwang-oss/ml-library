import numpy as np

class SoftmaxRegression():
    def __init__(self, learning_rate = 0.01, n_iters = 1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None 

    def fit(self, X, y):
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))
        k = len(np.unique(y))
        self.weights = np.zeros((X.shape[1], k))
        y_onehot = np.eye(k)[y]
        for _ in range(self.n_iters):
            prediction = X @ self.weights # dimensions
            predictionExp = np.exp(prediction)
            normalizedH = predictionExp / np.sum(predictionExp, axis = 1, keepdims = True)
            loss = -1 * np.sum(y_onehot * np.log(normalizedH)) / X.shape[0]
            if _ % 1000 == 0:
                print(loss)
            self.weights = self.weights - self.learning_rate * (1 / X.shape[0]) * X.T @ (normalizedH - y_onehot) # dimensions
        return self

    def predict(self, X):
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))
        hypothesis = X @ self.weights
        normalizedH = np.exp(hypothesis) / np.sum(np.exp(hypothesis), axis = 1, keepdims = True)
        return np.argmax(normalizedH, axis = 1)
    






            










