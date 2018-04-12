#import package pandas as pd
import pandas as pd
#import package numpy np
import numpy as np
#import Imputer from sklearn.preprocessing 
from sklearn.preprocessing import Imputer
#import train_test_split from sklearn.model_selection
from sklearn.model_selection import train_test_split
#import tree from sklearn
from sklearn import tree
#import accuracy_score
import accuracy_score
#import accuracy_score from sklearn.metrics
from sklearn.metrics import accuracy_score
#import confusion_matrixfrom sklearn.metrics
from sklearn.metrics import confusion_matrix
#import call from subprocess
from subprocess import call
# read breast-cancer.datathe dataset using method read_csv
df = pd.read_csv('breast-cancer.data',header=None)
# we are taking 10th coloumn using df_class
df_class = df[10]
df = df.drop(df.columns[10],axis=1)
#replace the data using replace method
df = df.replace('?',np.NaN)
#replace the all NaN missing values with column ean using Imputer method  
imp = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
#fitting the data 
imp.fit(df)
df_clean = imp.transform(df)
# taking coloumn list
column_list = ['Sample Code no.','Clump Thickness','Uniformity of cell size','Uniformity of cell shape',
               'Marginal adhesion','Single Epithelial cell size','Bare Nuclei','Bland Chromatin',
               'Normal Nucleoli','Mitoses']
# adding the names to coloumn list
df_clean = pd.DataFrame(np.array(df_clean),columns=column_list)
df_clean = df_clean.astype(int)
X = df_clean
y = df_class
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 1)
model = tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
#printing the accuracy score
print(accuracy_score(y_test,y_predict))
#printing the confusiion matrix 
print(confusion_matrix(y_test,y_predict))
tree.export_graphviz(model.tree_, out_file='tree.dot', feature_names=X.columns)
call(['dot','-T','png','tree.dot','-o','tree.png'])
