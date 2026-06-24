import numpy as np



def gini(y): 
    counts = np.bincount(y)
    p = counts / len(y)
    return 1 - np.sum(p**2)

def weighted_impurity(y_left, y_right):
    n = len(y_left) + len(y_right)
    return (len(y_left)/n) * gini(y_left) + (len(y_right)/n) * gini(y_right)

def majority_class(y):
    return np.bincount(y).argmax() 

def candidate_thresholds(col):
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
        if len(np.unique(y)) == 1:
            return Node(value = y[0])
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(value = majority_class(y))
        if len(y) < self.min_samples_split:
            return Node(value = majority_class(y))
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return Node(value = majority_class(y))
        mask = X[:, feature] < threshold 
        left = self._grow(X[mask], y[mask], depth + 1)
        right = self._grow(X[~mask], y[~mask], depth + 1)
        return Node(feature = feature, threshold = threshold, left = left, right = right)

    def _best_split(self, X, y):
        best = float('inf')
        bestFeature = None
        bestThreshold = None

        n_features = X.shape[1] # for ease of random forest implementation
        if self.max_features is None: 
            features_to_consider = range(n_features)
        else:
            features_to_consider = np.random.choice(n_features, self.max_features, replace = False)
        

        for feature in features_to_consider:
            thresholds = candidate_thresholds(X[:,feature])
            for threshold in thresholds:
                mask = X[:, feature] < threshold
                y_left = y[mask] 
                y_right = y[~mask]
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                if (impurity := weighted_impurity(y_left, y_right)) < best:
                    best = impurity
                    bestFeature = feature
                    bestThreshold = threshold
        return bestFeature, bestThreshold
    
    
    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])
    
    def _predict_one(self, x):
        node = self.root
        while not node.is_leaf():
            if x[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value
    


    
                
                



    

    