import numpy as np

from models.random_forest import RandomForest
from models.decision_trees import DecisionTree

rng = np.random.default_rng(0)
 
X0 = rng.normal(0, 1, size=(50, 2))
X1 = rng.normal(5, 1, size=(50, 2))
X = np.vstack([X0, X1])
y = np.array([0] * 50 + [1] * 50)
 
forest = RandomForest(n_trees=20, max_depth=3, max_features=1)
forest.fit(X, y)
preds = forest.predict(X)
print(np.mean(preds == y))