####################################################################
# Regression Machine Learning with Python                          #
# Similarity Methods                                               #
# (c) Diego Fernandez Garcia 2015-2018                             #
# www.exfinsis.com                                                 #
####################################################################

# 1. Packages Importing
import numpy as np
import pandas as pd
import sklearn.decomposition as fe
import sklearn.model_selection as cv
import sklearn.neighbors as ml
import sklearn.metrics as fa
import matplotlib.pyplot as plt
import time

#########

# 2. Data Reading
spy = pd.read_csv('Data//Regression-Machine-Learning-Data.txt', index_col='Date', parse_dates=True)

#########

# 3. Feature Creation

# 3.1. Target Feature
rspy = spy/spy.shift(1)-1
rspy.columns = ['rspy']

# 3.2. Predictor Features
rspy1 = rspy.shift(1)
rspy1.columns = ['rspy1']
rspy2 = rspy.shift(2)
rspy2.columns = ['rspy2']
rspy3 = rspy.shift(3)
rspy3.columns = ['rspy3']
rspy4 = rspy.shift(4)
rspy4.columns = ['rspy4']
rspy5 = rspy.shift(5)
rspy5.columns = ['rspy5']
rspy6 = rspy.shift(6)
rspy6.columns = ['rspy6']
rspy7 = rspy.shift(7)
rspy7.columns = ['rspy7']
rspy8 = rspy.shift(8)
rspy8.columns = ['rspy8']
rspy9 = rspy.shift(9)
rspy9.columns = ['rspy9']

# 3.3. All Features
rspyall = rspy
rspyall = rspyall.join(rspy1)
rspyall = rspyall.join(rspy2)
rspyall = rspyall.join(rspy3)
rspyall = rspyall.join(rspy4)
rspyall = rspyall.join(rspy5)
rspyall = rspyall.join(rspy6)
rspyall = rspyall.join(rspy7)
rspyall = rspyall.join(rspy8)
rspyall = rspyall.join(rspy9)
rspyall = rspyall.dropna()

#########

# 4. Range Delimiting

# 4.1. Training Range
rspyt = rspyall['2007-01-01':'2014-01-01']

# 4.2. Testing Range
rspyf = rspyall['2014-01-01':'2016-01-01']

#########

# 5. Algorithm Training and Testing

# 5.1. Similarity Methods

# 5.1.1. Algorithm Features

# Features Selection
pfa = ['rspy1', 'rspy2', 'rspy5']

# Features Extraction (Principal Component Analysis)
pfb = ['rspy1', 'rspy2', 'rspy3', 'rspy4', 'rspy5', 'rspy6', 'rspy7', 'rspy8', 'rspy9']
pcat = fe.PCA().fit_transform(rspyt[pfb], rspyt['rspy'])
pcaf = fe.PCA().fit_transform(rspyf[pfb], rspyf['rspy'])

# 5.1.2. K Nearest Neighbors KNN Regression

# 5.1.2.1 KNN Regression Optimal Parameter Selection

# Time Series Cross-Validation
# Exhaustive Grid Search Time Series Cross-Validation with Parameter Array Specification
# TimeSeriesSplit = anchored time series cross-validation with
# initial training subset = validating subset ~ n_samples / (n_splits + 1) in size
# n_neighbors = k nearest neighbors
cvknntsa = time.time()
cvknnta = cv.GridSearchCV(ml.KNeighborsRegressor(weights='uniform', algorithm='auto', metric='minkowski'),
                          cv=cv.TimeSeriesSplit(n_splits=4),
                          param_grid={'n_neighbors': [5, 6, 7]}).fit(rspyt[pfa], rspyt['rspy'])
cvknntea = time.time()
cvknntsb = time.time()
cvknntb = cv.GridSearchCV(ml.KNeighborsRegressor(weights='uniform', algorithm='auto', metric='minkowski'),
                          cv=cv.TimeSeriesSplit(n_splits=4),
                          param_grid={'n_neighbors': [5, 6, 7]}).fit(pcat, rspyt['rspy'])
cvknnteb = time.time()

# Time Series Cross-Validation Optimal Parameter Selection
cvknnpara = cvknnta.best_estimator_.n_neighbors
cvknnparb = cvknntb.best_estimator_.n_neighbors
print("")
print("== KNN Regression Optimal Parameter Selection ==")
print("")
print("KNN Regression A Optimal K Nearest Neighbors: ", cvknnpara)
print("KNN Regression B Optimal K Nearest Neighbors: ", cvknnparb)
print("")
print("KNN Regression A Training Time: ", cvknntea-cvknntsa, " seconds")
print("KNN Regression B Training Time: ", cvknnteb-cvknntsb, " seconds")
print("")

# 5.1.2.2. KNN Regression Algorithm Training
knnta = ml.KNeighborsRegressor(n_neighbors=cvknnpara, weights='uniform', algorithm='auto',
                               metric='minkowski').fit(rspyt[pfa], rspyt['rspy'])
knntb = ml.KNeighborsRegressor(n_neighbors=cvknnparb, weights='uniform', algorithm='auto',
                               metric='minkowski').fit(pcat, rspyt['rspy'])

# 5.1.2.3. KNN Regression Algorithm Testing
knnfa = knnta.predict(rspyf[pfa])
knnfb = knntb.predict(pcaf)

# 5.1.2.4. KNN Regression Algorithm Testing Charts

# KNN Regression A Testing Chart
knnfadf = pd.DataFrame(knnfa, index=rspyf.index)
fig1, ax = plt.subplots()
ax.plot(rspyf['rspy'])
ax.plot(knnfadf, label='knnfa')
plt.legend(loc='upper left')
plt.title('KNN Regression A Testing Chart')
plt.ylabel('Arithmetic Returns')
plt.xlabel('Date')
plt.show()

# KNN Regression B Testing Chart
knnfbdf = pd.DataFrame(knnfb, index=rspyf.index)
fig2, ax = plt.subplots()
ax.plot(rspyf['rspy'])
ax.plot(knnfbdf, label='knnfb')
plt.legend(loc='upper left')
plt.title('KNN Regression B Testing Chart')
plt.ylabel('Arithmetic Returns')
plt.xlabel('Date')
plt.show()

# 5.1.2.5. KNN Regression Algorithm Testing Forecasting Accuracy
knnmaea = fa.mean_absolute_error(rspyf['rspy'], knnfa)
knnmaeb = fa.mean_absolute_error(rspyf['rspy'], knnfb)
knnrmsea = np.sqrt(fa.mean_squared_error(rspyf['rspy'], knnfa))
knnrmseb = np.sqrt(fa.mean_squared_error(rspyf['rspy'], knnfb))
print("== KNN Regression Algorithm Testing Forecasting Accuracy ==")
print("")
print("KNN Regression A ", "MAE:", round(knnmaea, 6), "RMSE:", round(knnrmsea, 6))
print("KNN Regression B ", "MAE:", round(knnmaeb, 6), "RMSE:", round(knnrmseb, 6))