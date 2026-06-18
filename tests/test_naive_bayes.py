import numpy as np
from models.naive_bayes import NaiveBayes

X1 = (np.random.rand(50, 10) < 0.7).astype(int)
y1 = np.ones(50, dtype = int)

X0 = (np.random.rand(50, 10) < 0.3).astype(int)
y0 = np.zeros(50, dtype = int)

X = np.vstack((X1, X0))
y = np.concatenate((y1, y0))

model = NaiveBayes()

model.fit(X, y)

predictions = model.predict(X)

accuracy = np.mean(predictions == y)

print(accuracy)

assert accuracy > 0.9