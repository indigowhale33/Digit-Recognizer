# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 12:34:27 2016

@author: Steve Cho
"""

import pandas as pd
df_wine= pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header = None)

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
X, y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)

sc= StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.decomposition import PCA

pca = PCA(n_components = len(X.T))

X_train_pca_sklearn = pca.fit_transform(X_train_std)
variance_retained=0.9
cum_VE=0
i=1
while cum_VE < variance_retained:
    i=i+1
    cum_VE = sum(pca.explained_variance_ratio_[0:i])
    npcs=i
    

print ("Use", npcs, "principal components to retain ", variance_retained*100, "% of the variance")

pca = PCA(n_components = npcs)

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
 
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5, p=2, metric='minkowski')
knn.fit(X_train_pca, y_train)
y_pred = knn.predict(X_test_pca)