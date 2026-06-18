import numpy as np

class GDA():
    def __init__(self):
        self.phi = None
        self.mu_zero = None
        self.mu_one = None
        self.sigma = None

    def fit(self, X, y):
        self.phi = np.sum(y == 1) / len(y)
        self.mu_zero = np.sum(X[y == 0], axis = 0) / np.sum(y == 0)
        self.mu_one = np.sum(X[y == 1], axis = 0) / np.sum(y == 1)
        mu_matrix = np.where(y[:, None] == 0, self.mu_zero, self.mu_one)
        X_centered = X - mu_matrix
        self.sigma = (1 / len(y)) * X_centered.T @ X_centered
        return self
    
    def predict(self, X):
        sigma_inv = np.linalg.inv(self.sigma)
        log_det = np.log(np.linalg.det(self.sigma))

        diff1 = X - self.mu_one # m, n
        quad1 = np.sum((diff1 @ sigma_inv) * diff1, axis = 1) # m,
        log_px_y1 = -0.5 * log_det - 0.5 * quad1
        class1_scores = log_px_y1 + np.log(self.phi)

        diff0 = X - self.mu_zero
        quad0 = np.sum((diff0 @ sigma_inv) * diff0, axis = 1)
        log_px_y0 = -0.5 * log_det - 0.5* quad0
        class0_scores = log_px_y0 + np.log(1 - self.phi)
        return (class1_scores > class0_scores).astype(int)






