
#Project-4
#Example data has been downloaded from the open access Human Gene Expression Atlas and represents typical data bioinformaticians work with.


#It is "Transcription profiling by array of brain in humans, chimpanzees and macaques, and brain, heart, kidney and liver in orangutans" experiment in a tab-separated format.


#import all the required packages
#importing numpy package 
import numpy as np

import matplotlib.pyplot as plt

#from scipy.spatial.distance import pdist, squareform

from scipy.cluster.hierarchy import dendrogram

from fastcluster import *
#load data which is in textfile format columes from 1 to 31, data type as float and delimiter as new line
data = np.genfromtxt("ExpRawData-E-TABM-84-A-AFFY-44.tab",names=True,usecols=tuple(range(1,32)),dtype=float, delimiter="\t")


#printing length of data
print (len(data))

#printing length of data.dtype.names
print (len(data.dtype.names))

#printing labels data.dtype.names 
print (data.dtype.names)
#creating a new view of data 
data_array = data.view((np.float, len(data.dtype.names)))

#transposing the data_array matrix
data_array = data_array.transpose()


#printing data_array
print (data_array)


#data_dist = pdist(data_array) 
# computing the distance

data_link = linkage(data_array,method="single",metric="euclidean") 
print (data_link)
# computing the linkage

#creating a dendrogram with lables data.dtype.names   

dendrogram(data_link,labels=data.dtype.names)

#plotting X-axis name as samples
plt.xlabel('Samples')

#plotting Y-axis name as distance
plt.ylabel('Distance')

#title bottom of clustering with fontwieght and fontsize
plt.title('Bottom-up clustering', fontweight='bold', fontsize=14);

#display the plot
plt.show()


