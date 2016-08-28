# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 12:10:28 2016

@author: Steve
"""

import numpy as np
from scipy.special import expit
import pandas as pd

X = pd.read_csv('test.csv')
n_hidden = 30
n_features = X.shape[1]
n_output = 10

w1 = np.random.uniform(-1.0, 1.0, size=n_hidden*(n_features+1))
w1 = w1.reshape(n_hidden, n_features+1)
w2 = np.random.uniform(-1.0, 1.0, size = n_output*(n_hidden + 1))
w2 = w2.reshape(n_output, n_hidden+1)

X_new = np.ones((X.shape[0], X.shape[1]+1))

X_new[:,1:] = X
a1 = X_new
z2 = w1.dot(a1.T)
a2 = expit(z2)

a2_new = np.ones((a2.shape[0]+1, a2.shape[1]))
a2_new[1:,:]= a2
z3  = w2.dot(a2_new)
a3 = expit(z3)

y_pred = np.argmax(a3, axis=0)
np.savetxt("MNISTrandomized1.csv", y_pred, delimiter=',') 