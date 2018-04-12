import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from subprocess import call
df = pd.read_csv('breast-cancer.data',header=None)
df_class = df[10]
df = df.drop(df.columns[10],axis=1)
df = df.replace('?',np.NaN)
imp = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
imp.fit(df)
df_clean = imp.transform(df)
column_list = ['Sample Code no.','Clump Thickness','Uniformity of cell size','Uniformity of cell shape',
               'Marginal adhesion','Single Epithelial cell size','Bare Nuclei','Bland Chromatin',
               'Normal Nucleoli','Mitoses']
df_clean = pd.DataFrame(np.array(df_clean),columns=column_list)
df_clean = df_clean.astype(int)
X = df_clean
y = df_class
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 1)
model = tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
print(accuracy_score(y_test,y_predict))
print(confusion_matrix(y_test,y_predict))
tree.export_graphviz(model.tree_, out_file='tree.dot', feature_names=X.columns)
call(['dot','-T','png','tree.dot','-o','tree.png'])
