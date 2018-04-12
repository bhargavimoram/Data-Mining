

import numpy as np
import pandas as pd
from scipy.spatial import distance

data = pd.read_csv('ionosphere.data', header=None)  # Reading the file
print('Rows,Columns ', data.shape)    # Shape of the data

data_new = data.iloc[:,:-1]     # Removing the last column

rows = 351
total = rows * rows
# Create matrix
data_euc = np.arange(total,dtype=np.float).reshape(rows,rows)   
data_city = np.arange(total,dtype=np.float).reshape(rows,rows)
data_mink = np.arange(total,dtype=np.float).reshape(rows,rows)
data_inf = np.arange(total,dtype=np.float).reshape(rows,rows)
data_cos = np.arange(total,dtype=np.float).reshape(rows,rows)
data_jac = np.arange(total,dtype=np.float).reshape(rows,rows)
data = np.matrix(data_new)                                      
data = data.transpose()                                         
# Euclidean Distance
for i in range(34):
    for j in range(34):
        data_euc[i,j] = (distance.euclidean(data[i], data[j]))  

np.set_printoptions(precision=3)
print('Euclidean distance', data_euc)
# Manhattan Distance
for i in range(34):
    for j in range(34):
        data_city[i,j] = (distance.cityblock(data[i], data[j])) 
# Minkowski Distance
print('Manhattan distance', data_city)
for i in range(34):
    for j in range(34):
        data_mink[i,j] = (distance.minkowski(data[i], data[j], 3))  

print('Minkowski distance', data_mink)
# Chebyshev
for i in range(34):
    for j in range(34):
        data_inf[i,j] = (distance.chebyshev(data[i], data[j]))

print('Chebyshev  distance', data_inf)                   
# Cosine Similarity
for i in range(34):
    for j in range(34):
        data_cos[i,j] = (distance.cosine(data[i], data[j]))         

print('Cosine similarity', data_cos)
# Jaggard Similarity
for i in range(34):
    for j in range(34):
        data_jac[i,j] = (distance.jaccard(data[i], data[j]))        

print('Jaggard similarity', data_jac)


