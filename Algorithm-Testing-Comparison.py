####################################################################
# Regression Machine Learning with Python                          #
# Algorithm Testing Comparison                                     #
# (c) Diego Fernandez Garcia 2015-2018                             #
# www.exfinsis.com                                                 #
####################################################################

# 1. Packages Importing
import numpy as np
import pandas as pd
import sklearn.decomposition as fe
import sklearn.model_selection as cv
import sklearn.linear_model as lm
import sklearn.neighbors as knn
import sklearn.tree as tr
import sklearn.ensemble as en
import sklearn.svm as svm
import sklearn.neural_network as mlp
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

# 5.1. Algorithm Features

# Features Selection
pfa = ['rspy1', 'rspy2', 'rspy5']

# Features Extraction (Principal Component Analysis)
pfb = ['rspy1', 'rspy2', 'rspy3', 'rspy4', 'rspy5', 'rspy6', 'rspy7', 'rspy8', 'rspy9']
pcat = fe.PCA().fit_transform(rspyt[pfb], rspyt['rspy'])
pcaf = fe.PCA().fit_transform(rspyf[pfb], rspyf['rspy'])

# 5.2. Generalized Linear Models

# 5.2.1. Linear Regression

# 5.2.1.1. Linear Regression Algorithm Training
lmta = lm.LinearRegression().fit(rspyt[pfa], rspyt['rspy'])

# 5.2.1.2. Linear Regression Algorithm Testing
lmfa = lmta.predict(rspyf[pfa])

# 5.2.1.3. Linear Regression Algorithm Testing Forecasting Accuracy
lmmaea = fa.mean_absolute_error(rspyf['rspy'], lmfa)
lmrmsea = np.sqrt(fa.mean_squared_error(rspyf['rspy'], lmfa))

# 5.2.2. Ridge Regression

# 5.2.2.1 Ridge Regression Optimal Parameter Selection

# Time Series Cross-Validation
cvridgeta = cv.GridSearchCV(lm.Ridge(solver='auto'), cv=cv.TimeSeriesSplit(n_splits=4),
                          param_grid={'alpha': [0.01, 0.1, 1.0]}).fit(rspyt[pfa], rspyt['rspy'])

# Time Series Cross-Validation Optimal Parameter Selection
cvridgepara = cvridgeta.best_estimator_.alpha

# 5.2.2.2. Ridge Regression Algorithm Training
ridgeta = lm.Ridge(solver='auto', alpha=cvridgepara).fit(rspyt[pfa], rspyt['rspy'])

# 5.2.2.3. Ridge Regression Algorithm Testing
ridgefa = ridgeta.predict(rspyf[pfa])

# 5.2.2.4. Ridge Regression Algorithm Testing Forecasting Accuracy
ridgemaea = fa.mean_absolute_error(rspyf['rspy'], ridgefa)
ridgermsea = np.sqrt(fa.mean_squared_error(rspyf['rspy'], ridgefa))

# 5.3. Similarity Methods

# 5.3.1. K Nearest Neighbors KNN Regression

# 5.3.1.1 KNN Regression Optimal Parameter Selection

# Time Series Cross-Validation
cvknntb = cv.GridSearchCV(knn.KNeighborsRegressor(weights='uniform', algorithm='auto', metric='minkowski'),
                          cv=cv.TimeSeriesSplit(n_splits=4),
                          param_grid={'n_neighbors': [5, 6, 7]}).fit(pcat, rspyt['rspy'])

# Time Series Cross-Validation Optimal Parameter Selection
cvknnparb = cvknntb.best_estimator_.n_neighbors

# 5.3.1.2. KNN Regression Algorithm Training
knntb = knn.KNeighborsRegressor(n_neighbors=cvknnparb, weights='uniform', algorithm='auto',
                               metric='minkowski').fit(pcat, rspyt['rspy'])

# 5.3.1.3. KNN Regression Algorithm Testing
knnfb = knntb.predict(pcaf)

# 5.3.1.4. KNN Regression Algorithm Testing Forecasting Accuracy
knnmaeb = fa.mean_absolute_error(rspyf['rspy'], knnfb)
knnrmseb = np.sqrt(fa.mean_squared_error(rspyf['rspy'], knnfb))

# 5.4. Frequency Methods

# 5.4.1. Decision Tree DT Regression

# 5.4.1.1 DT Regression Optimal Parameter Selection

# Time Series Cross-Validation
cvdttb = cv.GridSearchCV(tr.DecisionTreeRegressor(criterion='mse', splitter='best'),
                          cv=cv.TimeSeriesSplit(n_splits=4),
                          param_grid={'max_depth': [1, 2, 3]}).fit(pcat, rspyt['rspy'])

# Time Series Cross-Validation Optimal Parameter Selection
cvdtparb = cvdttb.best_estimator_.max_depth

# 5.4.1.2. DT Regression Algorithm Training
dttb = tr.DecisionTreeRegressor(max_depth=cvdtparb, criterion='mse', splitter='best').fit(pcat, rspyt['rspy'])

# 5.4.1.3. DT Regression Algorithm Testing
dtfb = dttb.predict(pcaf)

# 5.4.1.4. DT Regression Algorithm Testing Forecasting Accuracy
dtmaeb = fa.mean_absolute_error(rspyf['rspy'], dtfb)
dtrmseb = np.sqrt(fa.mean_squared_error(rspyf['rspy'], dtfb))

# 5.5. Ensemble Methods

# 5.5.1. Random Forest RF Regression

# 5.5.1.1 RF Regression Optimal Parameter Selection

# Time Series Cross-Validation
cvrftb = cv.GridSearchCV(en.RandomForestRegressor(n_estimators=10, criterion='mse', bootstrap=True),
                          cv=cv.TimeSeriesSplit(n_splits=4),
                          param_grid={'max_depth': [1, 2, 3]}).fit(pcat, rspyt['rspy'])

# Time Series Cross-Validation Optimal Parameter Selection
cvrfparb = cvrftb.best_estimator_.max_depth

# 5.5.1.2. RF Regression Algorithm Training
rftb = en.RandomForestRegressor(max_depth=cvrfparb, n_estimators=10, criterion='mse', bootstrap=True).fit(pcat,
                                                                                                          rspyt['rspy'])

# 5.5.1.3. RF Regression Algorithm Testing
rffb = rftb.predict(pcaf)

# 5.5.1.4. RF Regression Algorithm Testing Forecasting Accuracy
rfmaeb = fa.mean_absolute_error(rspyf['rspy'], rffb)
rfrmseb = np.sqrt(fa.mean_squared_error(rspyf['rspy'], rffb))

# 5.5.2. Gradient Boosting Machine GBM Regression

# 5.5.2.1 GBM Regression Optimal Parameter Selection

# Time Series Cross-Validation
cvgbmta = cv.GridSearchCV(en.GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100),
                          cv=cv.TimeSeriesSplit(n_splits=4),
                          param_grid={'max_depth': [1, 2, 3]}).fit(rspyt[pfa], rspyt['rspy'])

# Time Series Cross-Validation Optimal Parameter Selection
cvgbmpara = cvgbmta.best_estimator_.max_depth

# 5.5.2.2. GBM Regression Algorithm Training
gbmta = en.GradientBoostingRegressor(max_depth=cvgbmpara, loss='ls', learning_rate=0.1,
                                     n_estimators=100).fit(rspyt[pfa], rspyt['rspy'])

# 5.5.2.3. GBM Regression Algorithm Testing
gbmfa = gbmta.predict(rspyf[pfa])

# 5.5.2.4. GBM Regression Algorithm Testing Forecasting Accuracy
gbmmaea = fa.mean_absolute_error(rspyf['rspy'], gbmfa)
gbmrmsea = np.sqrt(fa.mean_squared_error(rspyf['rspy'], gbmfa))

# 5.6. Maximum Margin Methods

# 5.6.1. Linear Support Vector Machine SVM Regression

# 5.6.1.1 Linear SVM Regression Optimal Parameter Selection

# Time Series Cross-Validation
cvlsvmta = cv.GridSearchCV(svm.SVR(kernel='linear', epsilon=0.1), cv=cv.TimeSeriesSplit(n_splits=4),
                          param_grid={'C': [0.25, 0.5, 1.0]}).fit(rspyt[pfa], rspyt['rspy'])

# Time Series Cross-Validation Optimal Parameter Selection
cvlsvmpara = cvlsvmta.best_estimator_.C

# 5.6.1.2. Linear SVM Regression Algorithm Training
lsvmta = svm.SVR(C=cvlsvmpara, kernel='linear', epsilon=0.1).fit(rspyt[pfa], rspyt['rspy'])

# 5.6.1.3. Linear SVM Regression Algorithm Testing
lsvmfa = lsvmta.predict(rspyf[pfa])

# 5.6.1.4. Linear SVM Regression Algorithm Testing Forecasting Accuracy
lsvmmaea = fa.mean_absolute_error(rspyf['rspy'], lsvmfa)
lsvmrmsea = np.sqrt(fa.mean_squared_error(rspyf['rspy'], lsvmfa))

# 5.6.2. Radial Basis Function RBF Support Vector Machine SVM Regression

# 5.6.2.1 RBF SVM Regression Optimal Parameter Selection

# Time Series Cross-Validation
cvrsvmta = cv.GridSearchCV(svm.SVR(kernel='rbf', epsilon=0.1), cv=cv.TimeSeriesSplit(n_splits=4),
                          param_grid={'C': [0.25, 0.5, 1.0]}).fit(rspyt[pfa], rspyt['rspy'])

# Time Series Cross-Validation Optimal Parameter Selection
cvrsvmpara = cvrsvmta.best_estimator_.C

# 5.6.2.2. RBF SVM Regression Algorithm Training
rsvmta = svm.SVR(C=cvrsvmpara, kernel='rbf', epsilon=0.1).fit(rspyt[pfa], rspyt['rspy'])

# 5.6.2.3. RBF SVM Regression Algorithm Testing
rsvmfa = rsvmta.predict(rspyf[pfa])

# 5.6.2.4. RBF SVM Regression Algorithm Testing Forecasting Accuracy
rsvmmaea = fa.mean_absolute_error(rspyf['rspy'], rsvmfa)
rsvmrmsea = np.sqrt(fa.mean_squared_error(rspyf['rspy'], rsvmfa))

# 5.7. Multi Layer Perceptron Methods

# 5.7.1. Artificial Neural Network ANN Regression

# 5.7.1.1 ANN Regression Optimal Parameter Selection

# Time Series Cross-Validation
cvanntb = cv.GridSearchCV(mlp.MLPRegressor(activation='relu', solver='adam'), cv=cv.TimeSeriesSplit(n_splits=4),
                          param_grid={'alpha': [0.001, 0.010, 0.100]}).fit(pcat, rspyt['rspy'])

# Time Series Cross-Validation Optimal Parameter Selection
cvannparb = cvanntb.best_estimator_.alpha

# 5.7.1.2. ANN Regression Algorithm Training
anntb = mlp.MLPRegressor(alpha=cvannparb, activation='relu', solver='adam').fit(pcat, rspyt['rspy'])

# 5.7.1.3. ANN Regression Algorithm Testing
annfb = anntb.predict(pcaf)

# 5.7.1.4. ANN Regression Algorithm Testing Forecasting Accuracy
annmaeb = fa.mean_absolute_error(rspyf['rspy'], annfb)
annrmseb = np.sqrt(fa.mean_squared_error(rspyf['rspy'], annfb))

# 5.8. Algorithm Testing Forecasting Accuracy Comparison
print("== Algorithm Testing Forecasting Accuracy Comparison ==")
print("")
print("Linear Regression A     ", "MAE:", round(lmmaea, 6), "RMSE:", round(lmrmsea, 6))
print("Ridge Regression A      ", "MAE:", round(ridgemaea, 6), "RMSE:", round(ridgermsea, 6))
print("KNN Regression B        ", "MAE:", round(knnmaeb, 6), "RMSE:", round(knnrmseb, 6))
print("DT Regression B         ", "MAE:", round(dtmaeb, 6), "RMSE:", round(dtrmseb, 6))
print("RF Regression B         ", "MAE:", round(rfmaeb, 6), "RMSE:", round(rfrmseb, 6))
print("GBM Regression A        ", "MAE:", round(gbmmaea, 6), "RMSE:", round(gbmrmsea, 6))
print("Linear SVM Regression A ", "MAE:", round(lsvmmaea, 6), "RMSE:", round(lsvmrmsea, 6))
print("RBF SVM Regression A    ", "MAE:", round(rsvmmaea, 6), "RMSE:", round(rsvmrmsea, 6))
print("ANN Regression B        ", "MAE:", round(annmaeb, 6), "RMSE:", round(annrmseb, 6))