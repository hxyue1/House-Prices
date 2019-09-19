import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import skew

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('train.csv')

##Exploratory data anlysis

#Checking for NAs

total = df_train.isnull().sum().sort_values(ascending=False)
            ##Looks like the PoolQC, MiscFeature, Alley and Fence columns are basically useless
            ##Lot frontage nas can be replaced with zeros?

#Replacing lot frontage NaNs with zeros
df_train.loc[df_train['LotFrontage'].isna(),'LotFrontage'] = 0

#Looking at garage columns
garage = df_train[['GarageCond','GarageType','GarageYrBlt','GarageFinish']]
garage__na = df_train[df_train['GarageCond'].isna()]

#Dropping columns
df_train = df_train.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageCond','GarageType','GarageYrBlt','GarageFinish', 'GarageQual'],axis=1)


#Check for NaNs again
total = df_train.isnull().sum().sort_values(ascending=False)
na_cols = total[total>0].index

#dropping columns
df_train = df_train.drop(na_cols, axis=1)

#Checking for skewness
con_cols = ['LotFrontage','LotArea','BsmtFinSF1','BsmtFinSF2','BsmtUnSF','TotalBsmtSF',]
skews = pd.Series(skew(df_train.loc[:,con_cols]),index=con_cols)

#Getting dummies
df_train = pd.get_dummies(df_train)

#Setting up dataframes for training models
X = df_train.drop('SalePrice',axis=1)
Y = df_train['SalePrice']

#Simple linear regression 
ols = sm.OLS(Y,X).fit()
preds = ols.predict(X)
rmse = (np.average((preds-Y)**2))**0.5

##Regularised linear models
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor as GBR
import xgboost as xgb

model = Ridge(alpha=8)
model.fit(X,Y)
preds = model.predict(X)
rmse = (np.average((preds-Y)**2))**0.5

dtrain = xgb.Dmatrix(X,label=y)
xgb.fit(dtrain)