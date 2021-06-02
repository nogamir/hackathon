import numpy as np

D = np.full((1, 5), 0.2)
y = np.array([1, 1, -1, 1, 1])
yt = np.array([1, 1, 1, -1, -1])

eps = np.sum((((y + yt) == 0) * D))

wt = 0.5

D = D * np.exp(-wt * y * yt)
print(D)
D = D / np.sum(D)
print(D)