####################################################################
# Regression Machine Learning with Python                          #
# Maximum Margin Methods                                           #
# (c) Diego Fernandez Garcia 2015-2018                             #
# www.exfinsis.com                                                 #
####################################################################

# 1. Packages Importing
import numpy as np
import pandas as pd
import sklearn.decomposition as fe
import sklearn.model_selection as cv
import sklearn.svm as ml
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

# 5.1. Maximum Margin Methods

# 5.1.1. Algorithm Features

# Features Selection
pfa = ['rspy1', 'rspy2', 'rspy5']

# Features Extraction (Principal Component Analysis)
pfb = ['rspy1', 'rspy2', 'rspy3', 'rspy4', 'rspy5', 'rspy6', 'rspy7', 'rspy8', 'rspy9']
pcat = fe.PCA().fit_transform(rspyt[pfb], rspyt['rspy'])
pcaf = fe.PCA().fit_transform(rspyf[pfb], rspyf['rspy'])

# 5.1.2. Linear Support Vector Machine SVM Regression

# 5.1.2.1 Linear SVM Regression Optimal Parameter Selection

# Time Series Cross-Validation
# Exhaustive Grid Search Time Series Cross-Validation with Parameter Array Specification
# TimeSeriesSplit = anchored time series cross-validation with
# initial training subset = validating subset ~ n_samples / (n_splits + 1) in size
# C = error term penalty
cvlsvmtsa = time.time()
cvlsvmta = cv.GridSearchCV(ml.SVR(kernel='linear', epsilon=0.1), cv=cv.TimeSeriesSplit(n_splits=4),
                          param_grid={'C': [0.25, 0.5, 1.0]}).fit(rspyt[pfa], rspyt['rspy'])
cvlsvmtea = time.time()
cvlsvmtsb = time.time()
cvlsvmtb = cv.GridSearchCV(ml.SVR(kernel='linear', epsilon=0.1), cv=cv.TimeSeriesSplit(n_splits=4),
                          param_grid={'C': [0.25, 0.5, 1.0]}).fit(pcat, rspyt['rspy'])
cvlsvmteb = time.time()

# Time Series Cross-Validation Optimal Parameter Selection
cvlsvmpara = cvlsvmta.best_estimator_.C
cvlsvmparb = cvlsvmtb.best_estimator_.C
print("")
print("== Linear SVM Regression Optimal Parameter Selection ==")
print("")
print("Linear SVM Regression A Optimal Error Term Penalty: ", cvlsvmpara)
print("Linear SVM Regression B Optimal Error Term Penalty: ", cvlsvmparb)
print("")
print("Linear SVM Regression A Training Time: ", cvlsvmtea-cvlsvmtsa, " seconds")
print("Linear SVM Regression B Training Time: ", cvlsvmteb-cvlsvmtsb, " seconds")
print("")

# 5.1.2.2. Linear SVM Regression Algorithm Training
lsvmta = ml.SVR(C=cvlsvmpara, kernel='linear', epsilon=0.1).fit(rspyt[pfa], rspyt['rspy'])
lsvmtb = ml.SVR(C=cvlsvmparb, kernel='linear', epsilon=0.1).fit(pcat, rspyt['rspy'])

# 5.1.2.3. Linear SVM Regression Algorithm Testing
lsvmfa = lsvmta.predict(rspyf[pfa])
lsvmfb = lsvmtb.predict(pcaf)

# 5.1.2.4. Linear SVM Regression Algorithm Testing Charts

# Linear SVM Regression A Testing Chart
lsvmfadf = pd.DataFrame(lsvmfa, index=rspyf.index)
fig1, ax = plt.subplots()
ax.plot(rspyf['rspy'])
ax.plot(lsvmfadf, label='lsvmfa')
plt.legend(loc='upper left')
plt.title('Linear SVM Regression A Testing Chart')
plt.ylabel('Arithmetic Returns')
plt.xlabel('Date')
plt.show()

# Linear SVM Regression B Testing Chart
lsvmfbdf = pd.DataFrame(lsvmfb, index=rspyf.index)
fig2, ax = plt.subplots()
ax.plot(rspyf['rspy'])
ax.plot(lsvmfbdf, label='lsvmfb')
plt.legend(loc='upper left')
plt.title('Linear SVM Regression B Testing Chart')
plt.ylabel('Arithmetic Returns')
plt.xlabel('Date')
plt.show()

# 5.1.2.5. Linear SVM Regression Algorithm Testing Forecasting Accuracy
lsvmmaea = fa.mean_absolute_error(rspyf['rspy'], lsvmfa)
lsvmmaeb = fa.mean_absolute_error(rspyf['rspy'], lsvmfb)
lsvmrmsea = np.sqrt(fa.mean_squared_error(rspyf['rspy'], lsvmfa))
lsvmrmseb = np.sqrt(fa.mean_squared_error(rspyf['rspy'], lsvmfb))
print("== Linear SVM Regression Algorithm Testing Forecasting Accuracy ==")
print("")
print("Linear SVM Regression A ", "MAE:", round(lsvmmaea, 6), "RMSE:", round(lsvmrmsea, 6))
print("Linear SVM Regression B ", "MAE:", round(lsvmmaeb, 6), "RMSE:", round(lsvmrmseb, 6))

# 5.1.3. Radial Basis Function RBF Support Vector Machine SVM Regression

# 5.1.3.1 RBF SVM Regression Optimal Parameter Selection

# Time Series Cross-Validation
# Exhaustive Grid Search Time Series Cross-Validation with Parameter Array Specification
# TimeSeriesSplit = anchored time series cross-validation with
# initial training subset = validating subset ~ n_samples / (n_splits + 1) in size
# C = error term penalty
cvrsvmtsa = time.time()
cvrsvmta = cv.GridSearchCV(ml.SVR(kernel='rbf', epsilon=0.1), cv=cv.TimeSeriesSplit(n_splits=4),
                          param_grid={'C': [0.25, 0.5, 1.0]}).fit(rspyt[pfa], rspyt['rspy'])
cvrsvmtea = time.time()
cvrsvmtsb = time.time()
cvrsvmtb = cv.GridSearchCV(ml.SVR(kernel='rbf', epsilon=0.1), cv=cv.TimeSeriesSplit(n_splits=4),
                          param_grid={'C': [0.25, 0.5, 1.0]}).fit(pcat, rspyt['rspy'])
cvrsvmteb = time.time()

# Time Series Cross-Validation Optimal Parameter Selection
cvrsvmpara = cvrsvmta.best_estimator_.C
cvrsvmparb = cvrsvmtb.best_estimator_.C
print("")
print("== RBF SVM Regression Optimal Parameter Selection ==")
print("")
print("RBF SVM Regression A Optimal Error Term Penalty: ", cvrsvmpara)
print("RBF SVM Regression B Optimal Error Term Penalty: ", cvrsvmparb)
print("")
print("RBF SVM Regression A Training Time: ", cvrsvmtea-cvrsvmtsa, " seconds")
print("RBF SVM Regression B Training Time: ", cvrsvmteb-cvrsvmtsb, " seconds")
print("")

# 5.1.3.2. RBF SVM Regression Algorithm Training
rsvmta = ml.SVR(C=cvrsvmpara, kernel='rbf', epsilon=0.1).fit(rspyt[pfa], rspyt['rspy'])
rsvmtb = ml.SVR(C=cvrsvmparb, kernel='rbf', epsilon=0.1).fit(pcat, rspyt['rspy'])

# 5.1.3.3. RBF SVM Regression Algorithm Testing
rsvmfa = rsvmta.predict(rspyf[pfa])
rsvmfb = rsvmtb.predict(pcaf)

# 5.1.3.4. RBF SVM Regression Algorithm Testing Charts

# RBF SVM Regression A Testing Chart
rsvmfadf = pd.DataFrame(rsvmfa, index=rspyf.index)
fig3, ax = plt.subplots()
ax.plot(rspyf['rspy'])
ax.plot(rsvmfadf, label='rsvmfa')
plt.legend(loc='upper left')
plt.title('RBF SVM Regression A Testing Chart')
plt.ylabel('Arithmetic Returns')
plt.xlabel('Date')
plt.show()

# RBF SVM Regression B Testing Chart
rsvmfbdf = pd.DataFrame(rsvmfb, index=rspyf.index)
fig4, ax = plt.subplots()
ax.plot(rspyf['rspy'])
ax.plot(rsvmfbdf, label='rsvmfb')
plt.legend(loc='upper left')
plt.title('RBF SVM Regression B Testing Chart')
plt.ylabel('Arithmetic Returns')
plt.xlabel('Date')
plt.show()

# 5.1.2.5. RBF SVM Regression Algorithm Testing Forecasting Accuracy
rsvmmaea = fa.mean_absolute_error(rspyf['rspy'], rsvmfa)
rsvmmaeb = fa.mean_absolute_error(rspyf['rspy'], rsvmfb)
rsvmrmsea = np.sqrt(fa.mean_squared_error(rspyf['rspy'], rsvmfa))
rsvmrmseb = np.sqrt(fa.mean_squared_error(rspyf['rspy'], rsvmfb))
print("== RBF SVM Regression Algorithm Testing Forecasting Accuracy ==")
print("")
print("RBF SVM Regression A ", "MAE:", round(rsvmmaea, 6), "RMSE:", round(rsvmrmsea, 6))
print("RBF SVM Regression B ", "MAE:", round(rsvmmaeb, 6), "RMSE:", round(rsvmrmseb, 6))