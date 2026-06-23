import numpy as np



def gini(y): 
    counts = np.bincount(y)
    p = counts / len(y)
    return 1 - np.sum(p**2)

def majority_class(y):
    return np.bincount(y).argmax() 

def candidate_threshold(col):
    vals = np.unique(col) # puts unique values in sorted order
    return (vals[:-1] + vals[1:]) / 2 # midpoint of each consecutive pair (consider [10, 30, 50])

class Node:
    def __init__(self, feature = None, threshold = None, left = None, right = None, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None
    

class DecisionTree():
    def __init__(self, max_depth = None, min_samples_split = 2, max_features = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        self.root = self._grow(X, y, depth = 0)

    def _grow(self, X, y, depth):
        if np.unique(y) == 1:
            return self
        elif depth >= self.max_depth:
            return self
        elif len(y) < self.min_samples_split:
            return self
        else:

    def _best_split(self, X, y):
        
    

    