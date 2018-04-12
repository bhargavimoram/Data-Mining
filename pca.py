
#data mining assignment 3
#import the package pandas 
import pandas
#import numpy
import numpy
# read the arrhythimia dataset using method read_csv
data = pandas.read_csv('arrhythmia.data',header=None)
#replace the data using replace method
data = data.replace('?',numpy.NaN)
#import Imputer method
from sklearn.preprocessing import Imputer
#replace the all NaN missing values with column ean using Imputer method  
imp = Imputer(missing_values= 'NaN' , strategy='mean', axis = 0)
imp.fit(data)
data_clean = imp.transform(data)
# import PCA
from sklearn.decomposition import PCA
#apply PCA to cleaned data set
pca = PCA(n_components = 100)
pca.fit(data_clean)
data_reduced = pca.transform(data_clean)
print(pca.explained_variance_ )
print(pca.components_)
print(pca.explained_variance_ratio_.sum())

