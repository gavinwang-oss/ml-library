import numpy as np

from models.logistic_regression import LogisticRegression


X = np.random.rand(100, 1)
y = (X[:, 0] > 0.5).astype(int)

logModel = LogisticRegression(0.01, 10000)
logModel.fit(X, y)
predicted_labels = logModel.predict(X)

accuracy = np.mean(predicted_labels == y)

print(accuracy)

assert accuracy > 0.9


