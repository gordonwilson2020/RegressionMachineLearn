####################################################################
# Regression Machine Learning with Python                          #
# Generalized Linear Models                                        #
# (c) Diego Fernandez Garcia 2015-2018                             #
# www.exfinsis.com                                                 #
####################################################################

# 1. Packages Importing
import numpy as np
import pandas as pd
import sklearn.decomposition as fe
import sklearn.model_selection as cv
import sklearn.linear_model as ml
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

# 5.1. Generalized Linear Models

# 5.1.1. Algorithm Features

# Features Selection
pfa = ['rspy1', 'rspy2', 'rspy5']

# Features Extraction (Principal Component Analysis)
pfb = ['rspy1', 'rspy2', 'rspy3', 'rspy4', 'rspy5', 'rspy6', 'rspy7', 'rspy8', 'rspy9']
pcat = fe.PCA().fit_transform(rspyt[pfb], rspyt['rspy'])
pcaf = fe.PCA().fit_transform(rspyf[pfb], rspyf['rspy'])

# 5.1.2. Linear Regression

# 5.1.2.2. Linear Regression Algorithm Training
lmta = ml.LinearRegression().fit(rspyt[pfa], rspyt['rspy'])
lmtb = ml.LinearRegression().fit(pcat, rspyt['rspy'])

# 5.1.2.3. Linear Regression Algorithm Testing
lmfa = lmta.predict(rspyf[pfa])
lmfb = lmtb.predict(pcaf)

# 5.1.2.4. Linear Regression Algorithm Testing Charts

# Linear Regression A Testing Chart
lmfadf = pd.DataFrame(lmfa, index=rspyf.index)
fig1, ax = plt.subplots()
ax.plot(rspyf['rspy'])
ax.plot(lmfadf, label='lmfa')
plt.legend(loc='upper left')
plt.title('Linear Regression A Testing Chart')
plt.ylabel('Arithmetic Returns')
plt.xlabel('Date')
plt.show()

# Linear Regression B Testing Chart
lmfbdf = pd.DataFrame(lmfb, index=rspyf.index)
fig2, ax = plt.subplots()
ax.plot(rspyf['rspy'])
ax.plot(lmfbdf, label='lmfb')
plt.legend(loc='upper left')
plt.title('Linear Regression B Testing Chart')
plt.ylabel('Arithmetic Returns')
plt.xlabel('Date')
plt.show()

# 5.1.2.5. Linear Regression Algorithm Testing Forecasting Accuracy
lmmaea = fa.mean_absolute_error(rspyf['rspy'], lmfa)
lmmaeb = fa.mean_absolute_error(rspyf['rspy'], lmfb)
lmrmsea = np.sqrt(fa.mean_squared_error(rspyf['rspy'], lmfa))
lmrmseb = np.sqrt(fa.mean_squared_error(rspyf['rspy'], lmfb))

print("== Linear Regression Algorithm Testing Forecasting Accuracy ==")
print("")
print("Linear Regression A ", "MAE:", round(lmmaea, 6), "RMSE:", round(lmrmsea, 6))
print("Linear Regression B ", "MAE:", round(lmmaeb, 6), "RMSE:", round(lmrmseb, 6))

# 5.1.3. Ridge Regression

# 5.1.3.1 Ridge Regression Optimal Parameter Selection

# Time Series Cross-Validation
# Exhaustive Grid Search Time Series Cross-Validation with Parameter Array Specification
# TimeSeriesSplit = anchored time series cross-validation with
# initial training subset = validating subset ~ n_samples / (n_splits + 1) in size
# alpha = L2 penalization coefficient (alpha = 0 = Linear Regression)
cvridgetsa = time.time()
cvridgeta = cv.GridSearchCV(ml.Ridge(solver='auto'), cv=cv.TimeSeriesSplit(n_splits=4),
                          param_grid={'alpha': [0.01, 0.1, 1.0]}).fit(rspyt[pfa], rspyt['rspy'])
cvridgetea = time.time()
cvridgetsb = time.time()
cvridgetb = cv.GridSearchCV(ml.Ridge(solver='auto'), cv=cv.TimeSeriesSplit(n_splits=4),
                          param_grid={'alpha': [0.01, 0.1, 1.0]}).fit(pcat, rspyt['rspy'])
cvridgeteb = time.time()

# Time Series Cross-Validation Optimal Parameter Selection
cvridgepara = cvridgeta.best_estimator_.alpha
cvridgeparb = cvridgetb.best_estimator_.alpha
print("")
print("== Ridge Regression Optimal Parameter Selection ==")
print("")
print("Ridge Regression A Optimal L2 Penalization Coefficient: ", cvridgepara)
print("Ridge Regression B Optimal L2 Penalization Coefficient: ", cvridgeparb)
print("")
print("Ridge Regression A Training Time: ", cvridgetea-cvridgetsa, " seconds")
print("Ridge Regression B Training Time: ", cvridgeteb-cvridgetsb, " seconds")
print("")

# 5.1.3.2. Ridge Regression Algorithm Training
ridgeta = ml.Ridge(solver='auto', alpha=cvridgepara).fit(rspyt[pfa], rspyt['rspy'])
ridgetb = ml.Ridge(solver='auto', alpha=cvridgeparb).fit(pcat, rspyt['rspy'])

# 5.1.3.3. Ridge Regression Algorithm Testing
ridgefa = ridgeta.predict(rspyf[pfa])
ridgefb = ridgetb.predict(pcaf)

# 5.1.3.4. Ridge Regression Algorithm Testing Charts

# Ridge Regression A Testing Chart
ridgefadf = pd.DataFrame(ridgefa, index=rspyf.index)
fig3, ax = plt.subplots()
ax.plot(rspyf['rspy'])
ax.plot(ridgefadf, label='ridgefa')
plt.legend(loc='upper left')
plt.title('Ridge Regression A Testing Chart')
plt.ylabel('Arithmetic Returns')
plt.xlabel('Date')
plt.show()

# Ridge Regression B Testing Chart
ridgefbdf = pd.DataFrame(ridgefb, index=rspyf.index)
fig4, ax = plt.subplots()
ax.plot(rspyf['rspy'])
ax.plot(ridgefbdf, label='ridgefb')
plt.legend(loc='upper left')
plt.title('Ridge Regression B Testing Chart')
plt.ylabel('Arithmetic Returns')
plt.xlabel('Date')
plt.show()

# 5.1.3.5. Ridge Regression Algorithm Testing Forecasting Accuracy
ridgemaea = fa.mean_absolute_error(rspyf['rspy'], ridgefa)
ridgemaeb = fa.mean_absolute_error(rspyf['rspy'], ridgefb)
ridgermsea = np.sqrt(fa.mean_squared_error(rspyf['rspy'], ridgefa))
ridgermseb = np.sqrt(fa.mean_squared_error(rspyf['rspy'], ridgefb))
print("== Ridge Regression Algorithm Testing Forecasting Accuracy ==")
print("")
print("Ridge Regression A ", "MAE:", round(ridgemaea, 6), "RMSE:", round(ridgermsea, 6))
print("Ridge Regression B ", "MAE:", round(ridgemaeb, 6), "RMSE:", round(ridgermseb, 6))