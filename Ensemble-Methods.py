####################################################################
# Regression Machine Learning with Python                          #
# Ensemble Methods                                                 #
# (c) Diego Fernandez Garcia 2015-2018                             #
# www.exfinsis.com                                                 #
####################################################################

# 1. Packages Importing
import numpy as np
import pandas as pd
import sklearn.decomposition as fe
import sklearn.model_selection as cv
import sklearn.ensemble as ml
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

# 5.1. Ensemble Methods

# 5.1.1. Algorithm Features

# Features Selection
pfa = ['rspy1', 'rspy2', 'rspy5']

# Features Extraction (Principal Component Analysis)
pfb = ['rspy1', 'rspy2', 'rspy3', 'rspy4', 'rspy5', 'rspy6', 'rspy7', 'rspy8', 'rspy9']
pcat = fe.PCA().fit_transform(rspyt[pfb], rspyt['rspy'])
pcaf = fe.PCA().fit_transform(rspyf[pfb], rspyf['rspy'])

# 5.1.2. Random Forest RF Regression

# 5.1.2.1 RF Regression Optimal Parameter Selection

# Time Series Cross-Validation
# Exhaustive Grid Search Time Series Cross-Validation with Parameter Array Specification
# TimeSeriesSplit = anchored time series cross-validation with
# initial training subset = validating subset ~ n_samples / (n_splits + 1) in size
# max_depth = maximum tree depth
cvrftsa = time.time()
cvrfta = cv.GridSearchCV(ml.RandomForestRegressor(n_estimators=10, criterion='mse', bootstrap=True),
                          cv=cv.TimeSeriesSplit(n_splits=4),
                          param_grid={'max_depth': [1, 2, 3]}).fit(rspyt[pfa], rspyt['rspy'])
cvrftea = time.time()
cvrftsb = time.time()
cvrftb = cv.GridSearchCV(ml.RandomForestRegressor(n_estimators=10, criterion='mse', bootstrap=True),
                          cv=cv.TimeSeriesSplit(n_splits=4),
                          param_grid={'max_depth': [1, 2, 3]}).fit(pcat, rspyt['rspy'])
cvrfteb = time.time()

# Time Series Cross-Validation Optimal Parameter Selection
cvrfpara = cvrfta.best_estimator_.max_depth
cvrfparb = cvrftb.best_estimator_.max_depth
print("")
print("== RF Regression Optimal Parameter Selection ==")
print("")
print("RF Regression A Optimal Maximum Tree Depth: ", cvrfpara)
print("RF Regression B Optimal Maximum Tree Depth: ", cvrfparb)
print("")
print("RF Regression A Training Time: ", cvrftea-cvrftsa, " seconds")
print("RF Regression B Training Time: ", cvrfteb-cvrftsb, " seconds")
print("")

# 5.1.2.2. RF Regression Algorithm Training
rfta = ml.RandomForestRegressor(max_depth=cvrfpara, n_estimators=10, criterion='mse', bootstrap=True).fit(rspyt[pfa],
                                                                                                          rspyt['rspy'])
rftb = ml.RandomForestRegressor(max_depth=cvrfparb, n_estimators=10, criterion='mse', bootstrap=True).fit(pcat,
                                                                                                          rspyt['rspy'])

# 5.1.2.3. RF Regression Algorithm Testing
rffa = rfta.predict(rspyf[pfa])
rffb = rftb.predict(pcaf)

# 5.1.2.4. RF Regression Algorithm Testing Charts

# RF Regression A Testing Chart
rffadf = pd.DataFrame(rffa, index=rspyf.index)
fig1, ax = plt.subplots()
ax.plot(rspyf['rspy'])
ax.plot(rffadf, label='rffa')
plt.legend(loc='upper left')
plt.title('RF Regression A Testing Chart')
plt.ylabel('Arithmetic Returns')
plt.xlabel('Date')
plt.show()

# RF Regression B Testing Chart
rffbdf = pd.DataFrame(rffb, index=rspyf.index)
fig2, ax = plt.subplots()
ax.plot(rspyf['rspy'])
ax.plot(rffbdf, label='rffb')
plt.legend(loc='upper left')
plt.title('RF Regression B Testing Chart')
plt.ylabel('Arithmetic Returns')
plt.xlabel('Date')
plt.show()

# 5.1.2.5. RF Regression Algorithm Testing Forecasting Accuracy
rfmaea = fa.mean_absolute_error(rspyf['rspy'], rffa)
rfmaeb = fa.mean_absolute_error(rspyf['rspy'], rffb)
rfrmsea = np.sqrt(fa.mean_squared_error(rspyf['rspy'], rffa))
rfrmseb = np.sqrt(fa.mean_squared_error(rspyf['rspy'], rffb))
print("== RF Regression Algorithm Testing Forecasting Accuracy ==")
print("")
print("RF Regression A ", "MAE:", round(rfmaea, 6), "RMSE:", round(rfrmsea, 6))
print("RF Regression B ", "MAE:", round(rfmaeb, 6), "RMSE:", round(rfrmseb, 6))

# 5.1.3. Gradient Boosting Machine GBM Regression

# 5.1.3.1 GBM Regression Optimal Parameter Selection

# Time Series Cross-Validation
# Exhaustive Grid Search Time Series Cross-Validation with Parameter Array Specification
# TimeSeriesSplit = anchored time series cross-validation with
# initial training subset = validating subset ~ n_samples / (n_splits + 1) in size
# max_depth = maximum tree depth
cvgbmtsa = time.time()
cvgbmta = cv.GridSearchCV(ml.GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100),
                          cv=cv.TimeSeriesSplit(n_splits=4),
                          param_grid={'max_depth': [1, 2, 3]}).fit(rspyt[pfa], rspyt['rspy'])
cvgbmtea = time.time()
cvgbmtsb = time.time()
cvgbmtb = cv.GridSearchCV(ml.GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100),
                          cv=cv.TimeSeriesSplit(n_splits=4),
                          param_grid={'max_depth': [1, 2, 3]}).fit(pcat, rspyt['rspy'])
cvgbmteb = time.time()

# Time Series Cross-Validation Optimal Parameter Selection
cvgbmpara = cvgbmta.best_estimator_.max_depth
cvgbmparb = cvgbmtb.best_estimator_.max_depth
print("")
print("== GBM Regression Optimal Parameter Selection ==")
print("")
print("GBM Regression A Optimal Maximum Tree Depth: ", cvgbmpara)
print("GBM Regression B Optimal Maximum Tree Depth: ", cvgbmparb)
print("")
print("GBM Regression A Training Time: ", cvgbmtea-cvgbmtsa, " seconds")
print("GBM Regression B Training Time: ", cvgbmteb-cvgbmtsb, " seconds")
print("")

# 5.1.3.2. GBM Regression Algorithm Training
gbmta = ml.GradientBoostingRegressor(max_depth=cvrfpara, loss='ls', learning_rate=0.1, n_estimators=100).fit(rspyt[pfa],
                                                                                                          rspyt['rspy'])
gbmtb = ml.GradientBoostingRegressor(max_depth=cvrfparb, loss='ls', learning_rate=0.1, n_estimators=100).fit(pcat,
                                                                                                          rspyt['rspy'])

# 5.1.3.3. GBM Regression Algorithm Testing
gbmfa = gbmta.predict(rspyf[pfa])
gbmfb = gbmtb.predict(pcaf)

# 5.1.3.4. GBM Regression Algorithm Testing Charts

# GBM Regression A Testing Chart
gbmfadf = pd.DataFrame(gbmfa, index=rspyf.index)
fig3, ax = plt.subplots()
ax.plot(rspyf['rspy'])
ax.plot(gbmfadf, label='gbmfa')
plt.legend(loc='upper left')
plt.title('GBM Regression A Testing Chart')
plt.ylabel('Arithmetic Returns')
plt.xlabel('Date')
plt.show()

# GBM Regression B Testing Chart
gbmfbdf = pd.DataFrame(gbmfb, index=rspyf.index)
fig4, ax = plt.subplots()
ax.plot(rspyf['rspy'])
ax.plot(gbmfbdf, label='gbmfb')
plt.legend(loc='upper left')
plt.title('GBM Regression B Testing Chart')
plt.ylabel('Arithmetic Returns')
plt.xlabel('Date')
plt.show()

# 5.1.3.5. GBM Regression Algorithm Testing Forecasting Accuracy
gbmmaea = fa.mean_absolute_error(rspyf['rspy'], gbmfa)
gbmmaeb = fa.mean_absolute_error(rspyf['rspy'], gbmfb)
gbmrmsea = np.sqrt(fa.mean_squared_error(rspyf['rspy'], gbmfa))
gbmrmseb = np.sqrt(fa.mean_squared_error(rspyf['rspy'], gbmfb))
print("== GBM Regression Algorithm Testing Forecasting Accuracy ==")
print("")
print("GBM Regression A ", "MAE:", round(gbmmaea, 6), "RMSE:", round(gbmrmsea, 6))
print("GBM Regression B ", "MAE:", round(gbmmaeb, 6), "RMSE:", round(gbmrmseb, 6))