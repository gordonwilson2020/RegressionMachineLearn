####################################################################
# Regression Machine Learning with Python                          #
# Algorithm Features                                               #
# (c) Diego Fernandez Garcia 2015-2018                             #
# www.exfinsis.com                                                 #
####################################################################

# 1. Packages Importing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.regression.linear_model as rg
import statsmodels.tools.tools as ct
import sklearn.feature_selection as fs
import sklearn.linear_model as lm
import sklearn.decomposition as fe

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
print(rspyall)

#########

# 4. Range Delimiting

# 4.1. Training Range
rspyt = rspyall['2007-01-01':'2014-01-01']

#########

# 5. Predictor Features Selection

# 5.1. Predictor Features Selection (Stepwise Linear Regression)

# 5.1.1. Linear Regression Predictor Features Selection (Step 1)
rspyt = ct.add_constant(rspyt, prepend=True)
lmtapf = ['const', 'rspy1', 'rspy2', 'rspy3', 'rspy4', 'rspy5', 'rspy6', 'rspy7', 'rspy8', 'rspy9']
lmta = rg.OLS(rspyt['rspy'], rspyt[lmtapf], hasconst=bool).fit()
print("")
print("== Linear Regression Predictor Features Selection (Step 1) ==")
print("")
print(lmta.summary())
print("")

# 5.1.2. Linear Regression Predictor Features Selection (Step 2)
lmtbpf = ['const', 'rspy1', 'rspy2', 'rspy5']
lmtb = rg.OLS(rspyt['rspy'], rspyt[lmtbpf], hasconst=bool).fit()
print("")
print("== Linear Regression Predictor Features Selection (Step 2) ==")
print("")
print(lmtb.summary())
print("")

# 5.1.3. Linear Regression Predictor Features Selection Scatter Charts

# Linear Regression Predictor Features Selection Scatter Charts (Previous Day Returns)
fig1, ax = plt.subplots()
lmtb1 = np.polyfit(rspyt['rspy1'], rspyt['rspy'], deg=1)
ax.scatter(rspyt['rspy1'], rspyt['rspy'])
ax.plot(rspyt['rspy1'], lmtb1[0] * rspyt['rspy1'] + lmtb1[1], color='red')
ax.set_title('rspyt vs rspy1t')
ax.set_ylabel('rspyt')
ax.set_xlabel('rspy1t')
plt.show()

# Linear Regression Predictor Features Selection Scatter Charts (Second Previous Day Returns)
fig2, ax = plt.subplots()
lmtb2 = np.polyfit(rspyt['rspy2'], rspyt['rspy'], deg=1)
ax.scatter(rspyt['rspy2'], rspyt['rspy'])
ax.plot(rspyt['rspy2'], lmtb2[0] * rspyt['rspy2'] + lmtb2[1], color='red')
ax.set_title('rspyt vs rspy2t')
ax.set_ylabel('rspyt')
ax.set_xlabel('rspy2t')
plt.show()

# Linear Regression Predictor Features Selection Scatter Charts (Previous Week Returns)
fig3, ax = plt.subplots()
lmtb3 = np.polyfit(rspyt['rspy5'], rspyt['rspy'], deg=1)
ax.scatter(rspyt['rspy5'], rspyt['rspy'])
ax.plot(rspyt['rspy5'], lmtb3[0] * rspyt['rspy5'] + lmtb3[1], color='red')
ax.set_title('rspyt vs rspy5t')
ax.set_ylabel('rspyt')
ax.set_xlabel('rspy5t')
plt.show()

# 5.1.4. Predictor Features Correlation Matrix
crspypf = ['rspy1', 'rspy2', 'rspy3', 'rspy4', 'rspy5', 'rspy6', 'rspy7', 'rspy8', 'rspy9']
crspyt = rspyt[crspypf]
fig4, ax = plt.subplots()
cax = ax.matshow(crspyt.corr(), cmap="Blues")
fig4.colorbar(cax)
ax.set_xticklabels(['']+crspypf)
ax.set_yticklabels(['']+crspypf)
ax.set_title('Predictor Features Correlation Matrix')
plt.show()

#########

# 5.2. Predictor Features Selection (Filter Methods)

# 5.2.1. Predictor Features
pft = ['rspy1', 'rspy2', 'rspy3', 'rspy4', 'rspy5', 'rspy6', 'rspy7', 'rspy8', 'rspy9']
pfft = fs.f_regression(rspyt[pft], rspyt['rspy'])
print("== Predictor Features Regression F Scores ==")
print("")
print("['rspy1', 'rspy2', 'rspy3', 'rspy4', 'rspy5', 'rspy6', 'rspy7', 'rspy8', 'rspy9']")
np.set_printoptions(precision=2)
print(pfft[0])
print("")

# 5.2.2. False Discovery Rate FDR
# Benjamini-Hochberg Procedure
fdrt = fs.SelectFdr(score_func=fs.f_regression, alpha=0.05).fit(rspyt[pft], rspyt['rspy'])
print("== FDR Features Selection ==")
print("")
print("['rspy1', 'rspy2', 'rspy3', 'rspy4', 'rspy5', 'rspy6', 'rspy7', 'rspy8', 'rspy9']")
print(fdrt.get_support())
print("")

# 5.2.3. Family-Wise Error Rate FWE
# Bonferroni Procedure
fwet = fs.SelectFwe(score_func=fs.f_regression, alpha=0.05).fit(rspyt[pft], rspyt['rspy'])
print("== FWE Features Selection ==")
print("")
print("['rspy1', 'rspy2', 'rspy3', 'rspy4', 'rspy5', 'rspy6', 'rspy7', 'rspy8', 'rspy9']")
print(fwet.get_support())
print("")

#########

# 5.3. Predictor Features Selection (Wrapper Methods)

# 5.3.1. Recursive Feature Elimination RFE
rfet = fs.RFE(estimator=lm.LinearRegression(), n_features_to_select=3).fit(rspyt[pft], rspyt['rspy'])
print("")
print("== RFE Features Selection ==")
print("")
print("['rspy1', 'rspy2', 'rspy3', 'rspy4', 'rspy5', 'rspy6', 'rspy7', 'rspy8', 'rspy9']")
print(rfet.get_support())
print(rfet.ranking_)
print("")

#########

# 5.3. Predictor Features Selection (Embedded Methods)

# 5.3.1. Least Absolute Shrinkage and Selection Operator LASSO
lassot = fs.SelectFromModel(estimator=lm.Lasso(alpha=0.1)).fit(rspyt[pft], rspyt['rspy'])
print("")
print("== LASSO Fetaures Selection ==")
print("")
print("['rspy1', 'rspy2', 'rspy3', 'rspy4', 'rspy5', 'rspy6', 'rspy7', 'rspy8', 'rspy9']")
print(lassot.get_support())
print("")

#########

# 6. Predictor Features Extraction

# 6.1. Principal Component Analysis PCA
pcat = fe.PCA().fit(rspyt[pft], rspyt['rspy'])
print("== PCA Features Extraction Explained Variance ==")
print("")
print("['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7', 'pc8', 'pc9']")
np.set_printoptions(precision=4)
print(pcat.explained_variance_ratio_)
print("")

# 5.2. Principal Component Analysis Bar Chart
fig5, ax = plt.subplots()
ax.bar(x=list(range(1, 10)), height=pcat.explained_variance_ratio_)
ax.set_title('Principal Component Analysis Explained Variance')
ax.set_ylabel('Explained Variance')
ax.set_xlabel('Principal Component')
plt.show()