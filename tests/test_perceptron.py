import numpy as np
from models.perceptron import Perceptron

X = np.random.rand(100, 1)
y = (X[:, 0] > 0.5).astype(int)

perceptronModel = Perceptron()
perceptronModel.fit(X, y)
predicted_labels = perceptronModel.predict(X)

accuracy = np.mean(predicted_labels == y)

print(accuracy)

assert accuracy > 0.9

