import numpy as np
from models.decision_trees import DecisionTree

class RandomForest():
    def __init__(self, n_trees = 100, max_depth = None, min_samples_split = 2, max_features = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split 
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        n = X.shape[0]
        for _ in range(self.n_trees):
            idxs = np.random.choice(n, n, replace = True)
            X_sample, y_sample = X[idxs], y[idxs]
            tree = DecisionTree(max_depth = self.max_depth, min_samples_split = self.min_samples_split, max_features = self.max_features)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)


    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = tree_preds.T
        return np.array([np.bincount(votes).argmax() for votes in tree_preds])
    
