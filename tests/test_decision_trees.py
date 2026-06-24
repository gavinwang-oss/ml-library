import numpy as np
from models.decision_trees import DecisionTree, Node


rng = np.random.default_rng(0)

X0 = rng.normal(0, 1, size =(50,2)) 
X1 = rng.normal(5, 1, size = (50, 2))
X = np.vstack([X0, X1])
y = np.array([0]*50 + [1] * 50)

tree = DecisionTree(max_depth = 3)
tree.fit(X, y)
preds = tree.predict(X)
print(np.mean(preds == y))

