####################################################################
# Regression Machine Learning with Python                          #
# Multi Layer Perceptron Methods                                   #
# (c) Diego Fernandez Garcia 2015-2018                             #
# www.exfinsis.com                                                 #
####################################################################

# 1. Packages Importing
import numpy as np
import pandas as pd
import sklearn.decomposition as fe
import sklearn.model_selection as cv
import sklearn.neural_network as ml
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

# 5.1. Multi Layer Perceptron Methods

# 5.1.1. Algorithm Features

# Features Selection
pfa = ['rspy1', 'rspy2', 'rspy5']

# Features Extraction (Principal Component Analysis)
pfb = ['rspy1', 'rspy2', 'rspy3', 'rspy4', 'rspy5', 'rspy6', 'rspy7', 'rspy8', 'rspy9']
pcat = fe.PCA().fit_transform(rspyt[pfb], rspyt['rspy'])
pcaf = fe.PCA().fit_transform(rspyf[pfb], rspyf['rspy'])

# 5.1.2. Artificial Neural Network ANN Regression

# 5.1.2.1 ANN Regression Optimal Parameter Selection

# Time Series Cross-Validation
# Exhaustive Grid Search Time Series Cross-Validation with Parameter Array Specification
# TimeSeriesSplit = anchored time series cross-validation with
# initial training subset = validating subset ~ n_samples / (n_splits + 1) in size
# alpha = node connection weight decay L2 regularization
cvanntsa = time.time()
cvannta = cv.GridSearchCV(ml.MLPRegressor(activation='relu', solver='adam'), cv=cv.TimeSeriesSplit(n_splits=4),
                          param_grid={'alpha': [0.001, 0.010, 0.100]}).fit(rspyt[pfa], rspyt['rspy'])
cvanntea = time.time()
cvanntsb = time.time()
cvanntb = cv.GridSearchCV(ml.MLPRegressor(activation='relu', solver='adam'), cv=cv.TimeSeriesSplit(n_splits=4),
                          param_grid={'alpha': [0.001, 0.010, 0.100]}).fit(pcat, rspyt['rspy'])
cvannteb = time.time()

# Time Series Cross-Validation Optimal Parameter Selection
cvannpara = cvannta.best_estimator_.alpha
cvannparb = cvanntb.best_estimator_.alpha
print("")
print("== ANN Regression Optimal Parameter Selection ==")
print("")
print("ANN Regression A Optimal Weight Decay L2 Regularization: ", cvannpara)
print("ANN Regression B Optimal Weight Decay L2 Regularization: ", cvannparb)
print("")
print("ANN Regression A Training Time: ", cvanntea-cvanntsa, " seconds")
print("ANN Regression B Training Time: ", cvannteb-cvanntsb, " seconds")
print("")

# 5.1.2.2. ANN Regression Algorithm Training
annta = ml.MLPRegressor(alpha=cvannpara, activation='relu', solver='adam').fit(rspyt[pfa], rspyt['rspy'])
anntb = ml.MLPRegressor(alpha=cvannparb, activation='relu', solver='adam').fit(pcat, rspyt['rspy'])

# 5.1.2.3. ANN Regression Algorithm Testing
annfa = annta.predict(rspyf[pfa])
annfb = anntb.predict(pcaf)

# 5.1.2.4. ANN Regression Algorithm Testing Charts

# ANN Regression A Testing Chart
annfadf = pd.DataFrame(annfa, index=rspyf.index)
fig1, ax = plt.subplots()
ax.plot(rspyf['rspy'])
ax.plot(annfadf, label='annfa')
plt.legend(loc='upper left')
plt.title('ANN Regression A Testing Chart')
plt.ylabel('Arithmetic Returns')
plt.xlabel('Date')
plt.show()

# ANN Regression B Testing Chart
annfbdf = pd.DataFrame(annfb, index=rspyf.index)
fig2, ax = plt.subplots()
ax.plot(rspyf['rspy'])
ax.plot(annfbdf, label='annfb')
plt.legend(loc='upper left')
plt.title('ANN Regression B Testing Chart')
plt.ylabel('Arithmetic Returns')
plt.xlabel('Date')
plt.show()

# 5.1.2.5. ANN Regression Algorithm Testing Forecasting Accuracy
annmaea = fa.mean_absolute_error(rspyf['rspy'], annfa)
annmaeb = fa.mean_absolute_error(rspyf['rspy'], annfb)
annrmsea = np.sqrt(fa.mean_squared_error(rspyf['rspy'], annfa))
annrmseb = np.sqrt(fa.mean_squared_error(rspyf['rspy'], annfb))
print("== ANN Regression Algorithm Testing Forecasting Accuracy ==")
print("")
print("ANN Regression A ", "MAE:", round(annmaea, 6), "RMSE:", round(annrmsea, 6))
print("ANN Regression B ", "MAE:", round(annmaeb, 6), "RMSE:", round(annrmseb, 6))