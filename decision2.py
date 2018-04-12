import pandas as pd

df = pd.read_csv('titanic_original.csv')

print(df.head())


df = df[['pclass','sex','age','sibsp','parch','fare','survived']]

df = df.dropna()


df['sex'] = df['sex'].map({'male':0,'female':1})


print(df.head())



X = df.drop('survived',axis=1)

y = df['survived']


from sklearn.model_selection import train_test_split

 X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 1)


from sklearn import tree

model = tree.DecisionTreeClassifier()


print (model)

model.fit(X_train,y_train)


y_predict = model.predict(X_test)


from sklearn.metrics import accuracy_score

print 
accuracy_score(y_test,y_predict)


from sklearn.metrics import confusion_matrix

print 
confusion_matrix(y_test,y_predict)




from sklearn.metrics import confusion_matrix

print 
confusion_matrix(y_test,y_predict)


tree.export_graphviz(model.tree_, out_file='tree.dot', feature_names=X.columns)

from subprocess import call

call(['dot','-T','png','tree.dot','-o','tree.png'])


