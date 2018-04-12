
#import the package numpy 
import numpy as np

#import the package pandas
import pandas as pd

# read the ionosphere.data dataset using method read_csv
data = pd.read_csv('ionosphere.data', header=None)
#printout the dimesions of the dataset
print(data)
#removing the last colomn
data_new = data.iloc[:,:-34]
#printing the dataset after removing the last colomn
print(data_new)
# import KMeans from sklearn.cluster 
from sklearn.cluster import KMeans
 # converting the dataset to binary

data = data.replace('g', 1)

data = data.replace('b', 0)

#import confusion_matrix from sklearn.metrics
from sklearn.metrics import confusion_matrix
kmeans = KMeans(n_clusters=2).fit(data)
labels = kmeans.labels_
conf = confusion_matrix(data_new, labels)
#print out confusion_matrix
print(conf)
#calculate purity for your confusion matrix
p1 = max(8,182)
print(purity)
p2 =max(30,131)
print(purity2)



