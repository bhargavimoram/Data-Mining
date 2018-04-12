
# assignment 4
#import the package numpy 
import numpy as np
#import the package pandas
import pandas as pd
#import the package datasets from sklearn
from sklearn import datasets
#import the package KMeans from sklearn.cluster 
from sklearn.cluster import KMeans
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
import matplotlib.patches as mpatches
#import confusion_matrix from sklearn.metrics 
from sklearn.metrics import confusion_matrix
# read the data set
iris = datasets.load_iris()
X = iris.data
y = iris.target
#applying Kmeans(cluster=3) to iris data set
kmeans = KMeans(n_clusters=3).fit(X)
labels = kmeans.labels_
conf = confusion_matrix(y, labels)
#print out confusion matrix
print(conf)
#plot 2D data set with 3 different clusters
colormap = np.array(['pink', 'Black', 'brown'])
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.colors()
plt.scatter(X[:, 3], X[:, 0], c=colormap[labels], s=100)
pink = mpatches.Patch(color='pink', label='The Petal Width')
black = mpatches.Patch(color='black', label='The Sepal Length')
brown = mpatches.Patch(color='brown', label='True Values')
plt.legend(handles=[pink,black,brown])
plt.show()