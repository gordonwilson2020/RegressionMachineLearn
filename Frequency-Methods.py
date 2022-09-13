####################################################################
# Regression Machine Learning with Python                          #
# Frequency Methods                                                #
# (c) Diego Fernandez Garcia 2015-2018                             #
# www.exfinsis.com                                                 #
####################################################################

# 1. Packages Importing
import numpy as np
import pandas as pd
import sklearn.decomposition as fe
import sklearn.model_selection as cv
import sklearn.tree as ml
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

# 5.1. Frequency Methods

# 5.1.1. Algorithm Features

# Features Selection
pfa = ['rspy1', 'rspy2', 'rspy5']

# Features Extraction (Principal Component Analysis)
pfb = ['rspy1', 'rspy2', 'rspy3', 'rspy4', 'rspy5', 'rspy6', 'rspy7', 'rspy8', 'rspy9']
pcat = fe.PCA().fit_transform(rspyt[pfb], rspyt['rspy'])
pcaf = fe.PCA().fit_transform(rspyf[pfb], rspyf['rspy'])

# 5.1.2. Decision Tree DT Regression

# 5.1.2.1 DT Regression Optimal Parameter Selection

# Time Series Cross-Validation
# Exhaustive Grid Search Time Series Cross-Validation with Parameter Array Specification
# TimeSeriesSplit = anchored time series cross-validation with
# initial training subset = validating subset ~ n_samples / (n_splits + 1) in size
# max_depth = maximum tree depth
cvdttsa = time.time()
cvdtta = cv.GridSearchCV(ml.DecisionTreeRegressor(criterion='mse', splitter='best'),
                          cv=cv.TimeSeriesSplit(n_splits=4),
                          param_grid={'max_depth': [1, 2, 3]}).fit(rspyt[pfa], rspyt['rspy'])
cvdttea = time.time()
cvdttsb = time.time()
cvdttb = cv.GridSearchCV(ml.DecisionTreeRegressor(criterion='mse', splitter='best'),
                          cv=cv.TimeSeriesSplit(n_splits=4),
                          param_grid={'max_depth': [1, 2, 3]}).fit(pcat, rspyt['rspy'])
cvdtteb = time.time()

# Time Series Cross-Validation Optimal Parameter Selection
cvdtpara = cvdtta.best_estimator_.max_depth
cvdtparb = cvdttb.best_estimator_.max_depth
print("")
print("== DT Regression Optimal Parameter Selection ==")
print("")
print("DT Regression A Optimal Maximum Tree Depth: ", cvdtpara)
print("DT Regression B Optimal Maximum Tree Depth: ", cvdtparb)
print("")
print("DT Regression A Training Time: ", cvdttea-cvdttsa, " seconds")
print("DT Regression B Training Time: ", cvdtteb-cvdttsb, " seconds")
print("")

# 5.1.2.2. DT Regression Algorithm Training
dtta = ml.DecisionTreeRegressor(max_depth=cvdtpara, criterion='mse', splitter='best').fit(rspyt[pfa], rspyt['rspy'])
dttb = ml.DecisionTreeRegressor(max_depth=cvdtparb, criterion='mse', splitter='best').fit(pcat, rspyt['rspy'])

# 5.1.2.3. DT Regression Algorithm Testing
dtfa = dtta.predict(rspyf[pfa])
dtfb = dttb.predict(pcaf)

# 5.1.2.4. DT Regression Algorithm Testing Charts

# DT Regression A Testing Chart
dtfadf = pd.DataFrame(dtfa, index=rspyf.index)
fig1, ax = plt.subplots()
ax.plot(rspyf['rspy'])
ax.plot(dtfadf, label='dtfa')
plt.legend(loc='upper left')
plt.title('DT Regression A Testing Chart')
plt.ylabel('Arithmetic Returns')
plt.xlabel('Date')
plt.show()

# DT Regression B Testing Chart
dtfbdf = pd.DataFrame(dtfb, index=rspyf.index)
fig2, ax = plt.subplots()
ax.plot(rspyf['rspy'])
ax.plot(dtfbdf, label='dtfb')
plt.legend(loc='upper left')
plt.title('DT Regression B Testing Chart')
plt.ylabel('Arithmetic Returns')
plt.xlabel('Date')
plt.show()

# 5.1.2.5. DT Regression Algorithm Testing Forecasting Accuracy
dtmaea = fa.mean_absolute_error(rspyf['rspy'], dtfa)
dtmaeb = fa.mean_absolute_error(rspyf['rspy'], dtfb)
dtrmsea = np.sqrt(fa.mean_squared_error(rspyf['rspy'], dtfa))
dtrmseb = np.sqrt(fa.mean_squared_error(rspyf['rspy'], dtfb))
print("== DT Regression Algorithm Testing Forecasting Accuracy ==")
print("")
print("DT Regression A ", "MAE:", round(dtmaea, 6), "RMSE:", round(dtrmsea, 6))
print("DT Regression B ", "MAE:", round(dtmaeb, 6), "RMSE:", round(dtrmseb, 6))