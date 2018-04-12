
#import fetch_lfw_people from sklearn.datasets 
from sklearn.datasets import fetch_lfw_people
# import the package pandas as pd
import pandas as pd
#import the package numpy as np
import numpy as np
# import PCA from sklearn.decomposition
from sklearn.decomposition import PCA
# import SVC from sklearn.svm
from sklearn.svm import SVC
# import train_test_split from sklearn.model_selection 
from sklearn.model_selection import train_test_split
#import classification_report from sklearn.metrics
from sklearn.metrics import classification_report
#load the face dataset
faces = fetch_lfw_people(min_faces_per_person=60)   
n_samples, h, w = faces.images.shape
#printing the face names
print(faces.target_names)
#printing the face image                           
print(faces.images.shape) 
#splitting the whole data set to the training dataset and testing data set.                          
X = faces.data
n_features = faces.data.shape[1]                    
y = faces.target
target_names = faces.target_names
n_classes = target_names.shape[0] 
               
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)
 
#applying PCA to the original dataset for dimesion reduction           
pca = PCA(n_components=200)                                    
pca.fit(X_train)                                                
X_train_pca = pca.transform(X_train)                            
X_test_pca = pca.transform(X_test) 
# using the training data set to build kernel support vector classifier        
clf = SVC(kernel='linear')                                     
clf = clf.fit(X_train_pca,y_train)                              
y_pred = clf.predict(X_test_pca) 
#printing the evaluate model                                
print(classification_report(y_test,y_pred,target_names=faces.target_names)) 