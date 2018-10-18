#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 14:23:10 2018

@author: wuzheqi
"""

import pandas as pd
DS = pd.read_csv('data0.csv')
Y = DS[["C_INNOVOUT1","C_INNOVOUT2","C_INNOVOUT3","C_INNOVOUT4","C_INNOVOUT5","C_INNOVOUT6"]]

X = DS[["RevisedTeamNumber",
        #"B_COCRFAC1","B_COCRFAC2","B_COCRFAC3","B_COCRFAC4","B_COCRFAC5","B_COCRFAC6","B_COCRFAC7","B_COCRFAC8","B_COCRFAC9","B_COCRFAC10","B_COCRFAC11","B_COCRFAC12","B_COCRFAC13",
        "M1_COCRFAC1","M1_COCRFAC2","M1_COCRFAC3","M1_COCRFAC4","M1_COCRFAC5","M1_COCRFAC6","M1_COCRFAC7","M1_COCRFAC8","M1_COCRFAC9","M1_COCRFAC10","M1_COCRFAC11","M1_COCRFAC12","M1_COCRFAC13",
        #"M2_COCRFAC1","M2_COCRFAC2","M2_COCRFAC3","M2_COCRFAC4","M2_COCRFAC5","M2_COCRFAC6","M2_COCRFAC7","M2_COCRFAC8","M2_COCRFAC9","M2_COCRFAC10","M2_COCRFAC11","M2_COCRFAC12","M2_COCRFAC13",
        "E_COCRFAC1","E_COCRFAC2","E_COCRFAC3","E_COCRFAC4","E_COCRFAC5","E_COCRFAC6","E_COCRFAC7","E_COCRFAC8","E_COCRFAC9","E_COCRFAC10","E_COCRFAC11","E_COCRFAC12","E_COCRFAC13"]]

# deal with scaling and missing value
from sklearn import preprocessing
# deal with missing value

import numpy as np
from fancyimpute import IterativeImputer
XY_incomplete = pd.concat([X, Y], axis=1)
XY_incomplete = XY_incomplete[:179]
print(XY_incomplete)
df = XY_incomplete.copy()
# group by team and deal with missing values
df[["M1_COCRFAC1","M1_COCRFAC2","M1_COCRFAC3","M1_COCRFAC4","M1_COCRFAC5","M1_COCRFAC6","M1_COCRFAC7","M1_COCRFAC8","M1_COCRFAC9","M1_COCRFAC10","M1_COCRFAC11","M1_COCRFAC12","M1_COCRFAC13",
    "E_COCRFAC1","E_COCRFAC2","E_COCRFAC3","E_COCRFAC4","E_COCRFAC5","E_COCRFAC6","E_COCRFAC7","E_COCRFAC8","E_COCRFAC9","E_COCRFAC10","E_COCRFAC11","E_COCRFAC12","E_COCRFAC13",
    "C_INNOVOUT1","C_INNOVOUT2","C_INNOVOUT3","C_INNOVOUT4","C_INNOVOUT5","C_INNOVOUT6"
    ]] = df.groupby(['RevisedTeamNumber'])["M1_COCRFAC1","M1_COCRFAC2","M1_COCRFAC3","M1_COCRFAC4","M1_COCRFAC5","M1_COCRFAC6","M1_COCRFAC7","M1_COCRFAC8","M1_COCRFAC9","M1_COCRFAC10","M1_COCRFAC11","M1_COCRFAC12","M1_COCRFAC13",
    "E_COCRFAC1","E_COCRFAC2","E_COCRFAC3","E_COCRFAC4","E_COCRFAC5","E_COCRFAC6","E_COCRFAC7","E_COCRFAC8","E_COCRFAC9","E_COCRFAC10","E_COCRFAC11","E_COCRFAC12","E_COCRFAC13",
    "C_INNOVOUT1","C_INNOVOUT2","C_INNOVOUT3","C_INNOVOUT4","C_INNOVOUT5","C_INNOVOUT6"].transform(lambda x: x.fillna(x.mean()))

# fill the other missing value with column mean rather than MICE -- since the filling value will be out of range
'''
# MICE code
ME = df.copy()
n_imputations = 32
XY_completed = []
for i in range(n_imputations):
    imputer = IterativeImputer(n_iter=32, sample_posterior=True, random_state=i)
    #a = imputer.fit_transform(XY_incomplete)
    #print(imputer.fit_transform(XY_incomplete))
XY_completed.append(imputer.fit_transform(ME))

XY_completed_mean = np.mean(XY_completed, 0)
XY_completed_std = np.std(XY_completed, 0)
'''

df.fillna(df.mean(), inplace=True)
df = round(df)
# could add weights on Y1-Y6
Y = np.sum(df[["C_INNOVOUT1","C_INNOVOUT2","C_INNOVOUT3","C_INNOVOUT4","C_INNOVOUT5","C_INNOVOUT6"]],axis = 1)
y = pd.DataFrame(Y,columns=['Y'])#Y.values
import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(Y);
# create new features using polynomial expansion
# feature selection
X = df[["M1_COCRFAC1","M1_COCRFAC2","M1_COCRFAC3","M1_COCRFAC4","M1_COCRFAC5","M1_COCRFAC6","M1_COCRFAC7","M1_COCRFAC8","M1_COCRFAC9","M1_COCRFAC10","M1_COCRFAC11","M1_COCRFAC12","M1_COCRFAC13",
    "E_COCRFAC1","E_COCRFAC2","E_COCRFAC3","E_COCRFAC4","E_COCRFAC5","E_COCRFAC6","E_COCRFAC7","E_COCRFAC8","E_COCRFAC9","E_COCRFAC10","E_COCRFAC11","E_COCRFAC12","E_COCRFAC13",
    ]]
df1 = pd.concat([X, y], axis=1)
#df1.to_csv('train.csv', encoding='utf-8', index=False)

# eliminate redundant features
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
a = sel.fit_transform(X)

# create polynomial features
from sklearn.preprocessing import PolynomialFeatures
'''
poly7 = PolynomialFeatures(degree = 7) #interaction_only=True
a = poly7.fit_transform(X[["M1_COCRFAC1","M1_COCRFAC2","M1_COCRFAC3","M1_COCRFAC4","M1_COCRFAC5","M1_COCRFAC6","M1_COCRFAC7"]])
#poly.get_feature_names()
b = poly7.fit_transform(X[["E_COCRFAC1","E_COCRFAC2","E_COCRFAC3","E_COCRFAC4","E_COCRFAC5","E_COCRFAC6","E_COCRFAC7"]])
poly2 = PolynomialFeatures(degree = 2)
c = poly2.fit_transform(X[["M1_COCRFAC11","M1_COCRFAC12"]])
d = poly2.fit_transform(X[["E_COCRFAC11","E_COCRFAC12"]])

a = pd.DataFrame(a)
b = pd.DataFrame(b)
c = pd.DataFrame(c)
d = pd.DataFrame(d)
poly_X = pd.concat([a,b,c,d], axis=1)

df2 = pd.concat([X, poly_X], axis=1)
df2 = pd.concat([df2, y], axis=1)
#df2.to_csv('train.csv', encoding='utf-8', index=False)
'''
poly2 = PolynomialFeatures(degree = 2)
poly_X = poly2.fit_transform(X.names())
poly2.get_feature_names()
poly_X = pd.DataFrame(poly_X,columns = poly2.get_feature_names())
df3 = pd.concat([poly_X, y], axis=1)
df3.to_csv('train.csv', encoding='utf-8', index=False)




import matplotlib.pyplot as plt
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
rfecv.fit(X, y)
rfecv.ranking_
criteria = rfecv.support_ 
rfecv.estimator_
print("Optimal number of features : %d" % rfecv.n_features_)

X1 = X.iloc[:, criteria]

rfecv.get_support()
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

sorted(sklearn.metrics.SCORERS.keys())


# do feature engineering but do not apply any transformation(hard to explain)
from sklearn.feature_selection import RFECV
RFECV
