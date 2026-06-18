import numpy as np
from models.gaussian_discrimant_analysis import GDA

X1 = np.random.randn(50, 2) + np.array([2, 2])
y1 = np.ones(50, dtype = int)

X0 =np.random.randn(50, 2) + np.array([-2, -2])
y0 = np.zeros(50, dtype = int)

X = np.vstack((X1, X0)) 
y = np.concatenate((y1, y0))

model = GDA()

model.fit(X, y)

predictions = model.predict(X)

accuracy = np.mean(predictions == y)
print(accuracy)
assert(accuracy > 0.9)

