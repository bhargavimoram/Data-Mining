
# assignment 4
#import the package numpy 
import numpy as np

#import the package pandas
import pandas as pd

#import imputer method
from sklearn.preprocessing import Imputer
# import apriori and association_rules from mlxtend.frequesntpatterns
from mlxtend.frequent_patterns import apriori

from mlxtend.frequent_patterns import association_rules

# read the house-votes-84 dataset using method read_csv
data = pd.read_csv('house-votes-84.data', header=None)
# print the dimension of the data set
print('rows*columns', data.shape)

# converting the dataset to binary

data = data.replace('y', 1)

data = data.replace('n', 0)

data = data.replace('republican', 1)

data = data.replace('democrat', 0)

data = data.replace('?', np.NaN)

imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

imp.fit(data)

data_clean = imp.transform(data)

#Columns list
columns = ['republican','handicapped-infants','water-project-cost-sharing','adoptionof-the-budget-resolution',
               'physician-fee-freeze','el-salvador-aid','eligious-groups-inschools','anti-satellite-test-ban'
    ,'aid-to-nicaraguan-contras','mx-missile','immigration','synfuelscorporation-cutback',
    'education-spending','superfund-right-to-sue','crime','dutyfree-exports','export-administration-act-south-africa']
data1 = pd.DataFrame(np.array(data_clean), columns=columns)
data1['Democrat'] = data1.republican.apply(lambda x: 0 if x == 1 else 1)

#aplly apriori algorithm to data set with minsup = 0.3
freq_itemsets=apriori(data1,min_support=0.3,use_colnames=True)

#print frequent datasets
print(freq_itemsets)

data2 = pd.DataFrame(data=freq_itemsets)

#aplly apriori algorithm to data set with minsup = 0.9
ass_rules = association_rules(data2,metric="confidence", min_threshold=0.9)

print(ass_rules)