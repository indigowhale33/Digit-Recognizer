# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 02:19:36 2016

@author: Steve
"""
from sklearn import neighbors
import pandas as pd
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler

print ("reading training data")


reader = pd.read_csv('Problem7_train.csv')
X = []
Y = []

Y = reader["label"].values
del reader["label"]
X = reader[:].values

#for row in reader:
#   print(row[0:])
        #if reader.line_num > 1:
         #   X.append(row[1:])
          #  Y.append(row[0])

    #X = 784 x N, Y = N
#X = np.array(X)
Y = np.array(Y)

print(Y)
print(X)

print ("reading test data")

reader2 = pd.read_csv('Problem7_test.csv')
    
del reader2["ID"]
X_test = []
X_test = reader2.values

X_test = np.array(X_test)

print ("fitting model")

k_neighbors = 3
clf = neighbors.KNeighborsClassifier(k_neighbors, p = 2, metric = 'minkowski')

sc = StandardScaler()
sc.fit(X)
X_train_std = sc.transform(X)
X_test_std = sc.transform(X_test)

clf.fit(X_train_std, Y)

print ("making predictions")

Z = clf.predict(X_test_std)
#print(accuracy_score(X_,y_pred))

print ("writing predictions to file")

with open('submission.csv', 'w+') as f:
    header = 'ID,Label\n'
    f.write(header)
    Id = 0
    for prediction in Z:
        Id += 1
        line = str(Id) + ',' + str(prediction) + '\n'
        f.write(line)
        
print("done")