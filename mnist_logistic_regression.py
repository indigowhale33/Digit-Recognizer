# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 04:36:51 2016

@author: Steve
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd

reader = pd.read_csv('train.csv')
Y = reader["label"].values
del reader["label"]
X = reader[:].values
Y = np.array(Y)


print ("reading test data")

reader2 = pd.read_csv('test.csv')
    
#del reader2["ID"]
X_test = []
X_test = reader2.values

X_test = np.array(X_test)

sc = StandardScaler()
sc.fit(X)
X_train_std = sc.fit_transform(X)
X_test_std = sc.transform(X_test)

print('regression fit')

lr = LogisticRegression(C=10000, random_state=0)
print('regression fit C=10000')
lr.fit(X_train_std,Y)

#print(lr.predict_proba(X_test_std[0:]))

with open('c10000.csv', 'w+') as f:
    header = 'ImageId,Label\n'
    f.write(header)
    Id = 0
    for prediction in lr.predict(X_test_std[0:]):
        Id += 1
        line = str(Id) + ',' + str(prediction) + '\n'
        f.write(line)
        
print("done")