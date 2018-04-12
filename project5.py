import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score
from sklearn import metrics
tr = pd.read_csv('tr_server_data.csv', header=None)
cv = pd.read_csv('cv_server_data.csv', header=None)
gt = pd.read_csv('gt_server_data.csv', header=None)
print(tr)
print(cv)
print(gt)
mu = np.mean(tr, axis=0)
print(mu)
sigma = np.cov(tr.T) 
print(sigma)
pmodel = multivariate_normal(mean=mu, cov=sigma)
densityprobability = pmodel.pdf(tr)
print(densityprobability)
threshold = (min(densityprobability),max(densityprobability))
print(threshold)
y_true=tr.iloc[:,0]
y_pred=cv.iloc[:,0]
precision, recall, thresholds = precision_recall_curve(y_true,y_pred)