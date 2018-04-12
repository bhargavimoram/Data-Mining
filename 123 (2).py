
Project-1 Data mining"""

import numpy as np			# importing packages
import pandas as pd
from scipy.spatial import distance

data = pd.read_csv('ionosphere.data',header=None)	# read the data file "ionosphere"
print(data.shape)
data_new = data.iloc[:,:-1]				# remove last column from the data
rows = 351
total = rows * rows

data= np.matrix(data_new)				
data=data.transpose()					# transpose the matrix
print('New shape:', data.shape)
data_manhattan = np.arange(total,dtype=np.float).reshape(rows,rows)	# Manhattan distance
for x in range(34):
	for y in range(34):
		data_manhattan[x,y] = (distance.cityblock(data[x], data[y]))
print(data_manhattan)

data_euclidean = np.arange(total,dtype=np.float).reshape(rows,rows)	# Euclidean distance
for x in range(34):
	for y in range(34):
		data_euclidean[x,y] = (distance.euclidean(data[x], data[y]))
print(data_euclidean)

data_minkowski = np.arange(total,dtype=np.float).reshape(rows,rows)	# Minkowski distance
for x in range(34):
	for y in range(34):
		data_minkowski[x,y] = (distance.minkowski(data[x], data[y], 3))
print(data_minkowski)

data_chebyshev = np.arange(total,dtype=np.float).reshape(rows,rows)	# Chebyshev distance
for x in range(34):
	for y in range(34):
		data_chebyshev[x,y] = (distance.chebyshev(data[x], data[y]))
print(data_chebyshev)

data_cosine = np.arange(total,dtype=np.float).reshape(rows,rows)	# Cosine similarity
for x in range(34):
	for y in range(34):
		data_cosine[x,y] = (distance.cosine(data[x], data[y]))
print(data_cosine)

data_jaccard = np.arange(total,dtype=np.float).reshape(rows,rows)	# Jaccard similariy
for x in range(34):	
	for y in range(34):
		data_jaccard[x,y] = (distance.jaccard(data[x], data[y]))
print(data_jaccard)