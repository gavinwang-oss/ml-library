import numpy as np

class NaiveBayes():
    def __init__(self):
        self.phi = None
        self.prob_word_given_1 = None
        self.prob_word_given_0 = None
    

    def fit(self, X, y):
        self.phi = np.sum(y == 1) / len(y)
        c1rows = X[y == 1]
        word_counts_c1 = np.sum(c1rows, axis = 0) # 1 x n
        self.prob_word_given_1 = (word_counts_c1 + 1) / (np.sum(y == 1) + 2)
        c0rows = X[y == 0]
        word_counts_c0 = np.sum(c0rows, axis = 0) 
        self.prob_word_given_0 = (word_counts_c0 + 1) / (np.sum(y == 0) + 2)
        return self

    def predict(self, X):
        log_probs_1 = X * np.log(self.prob_word_given_1) + (1 - X) * np.log(1 - self.prob_word_given_1)
        class1_scores = np.sum(log_probs_1, axis = 1)
        class1_scores = class1_scores + np.log(self.phi)
        log_probs_0 = X * np.log(self.prob_word_given_0) + (1 - X) * np.log(1 - self.prob_word_given_0)
        class0_scores = np.sum(log_probs_0, axis = 1)
        class0_scores = class0_scores + np.log(1 - self.phi)
        return (class1_scores > class0_scores).astype(int)
