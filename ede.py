import numpy as np
import math
from scipy.spatial import distance
X1 = [12, 14, 18, 23, 27, 28, 34, 37, 39, 40]
Y1 = [300, 500, 1000, 2000, 3500, 4000, 4300, 6000, 2500, 2700]
data = np.matrix([X1,Y1])

data = data.transpose()
print(data)

print(data.shape)
print(data[1])
print(data[0].item(0))
for i in range  (10):
for j in range  (10):

data_eclidean[i,j] = distance.euclidean (data[i], data[j]))
print(data_eclidean)
