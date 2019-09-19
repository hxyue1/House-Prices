#https://www.kaggle.com/apapiu/regularized-linear-models/
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from scipy.stats import jarque_bera

#######Importing data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],test.loc[:,'MSSubClass':'SaleCondition']))
#####################



#######Data exploration

#Checking for nas
na_count = all_data.isnull().sum().sort_values(ascending=False)

#Biggest culprits
trouble_cols = list(all_data.isnull().sum().sort_values(ascending=False).index[0:7])

#First four are particularly bad, but the other variables are salvageable

#Dropping first four problem variables 
all_data = all_data.drop(trouble_cols[0:4],axis=1)

#Replacing missing fire place quality with NFP (no fire place)
all_data['FireplaceQu']=all_data['FireplaceQu'].fillna('NFP')

#Replacing missing LotFrontage values with zero, these are probably houses with no frontage
all_data['LotFrontage']=all_data['LotFrontage'].fillna(0)

#Dealing with the problem garage variables
garage_trouble_cols = ['GarageCond','GarageQual','GarageYrBlt','GarageFinish','GarageType']
sns.countplot(x='GarageCond',data=all_data)
sns.countplot(x='GarageQual',data=all_data)
sns.countplot(x='GarageYrBlt',data=all_data)
sns.countplot(x='GarageFinish',data=all_data)
sns.countplot(x='GarageType',data=all_data)

#I suspect that missing garage values are all because the house doesn't actually have a garage
garage_trouble = all_data.loc[:,garage_trouble_cols]
garage_trouble[garage_trouble['GarageCond'].isnull()] #Yeah, looks like it

#Replacing missing categorical garage data with NG (no garage)
garage_cat_missing = ['GarageCond','GarageQual','GarageFinish','GarageType']
all_data.loc[:,garage_trouble_cols]=all_data.loc[:,garage_trouble_cols].fillna('NG')


#######Transforming data
#numeric feats
numeric_feats_cols = ['LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YrSold']

#log transform skewed numeric functions:
skewed_feats = train[numeric_feats_cols].apply(lambda x:skew(x.dropna()))
n_train = len(train.index)
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

#Getting dummies
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str) #MSSubClass is a categorical variable, so we convert it to a string type so pd.get_dummies recognises it as such
all_data = pd.get_dummies(all_data)

#Filling nas with mean
all_data = all_data.fillna(all_data.mean())

#Matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = np.log1p(train.SalePrice)
#####################


##Setting up training procedures
from sklearn.model_selection import cross_val_score
from scipy.optimize import minimize_scalar

def rmse_lasso(lambda2):
    rmse = (-cross_val_score(Lasso(alpha=lambda2), X_train, y, scoring='neg_mean_squared_error', cv=5))**0.5
    rmse = rmse.mean()
    return(rmse)


##########XGBOOOOOOOST

import xgboost as xgb

xg_train = xgb.DMatrix(X_train, label=y)
xg_test = xgb.DMatrix(X_test)

def rmse_cv(model):
    rmse = (-cross_val_score(model, X_train, y, scoring='neg_mean_squared_error', cv=5))**0.5
    rmse = rmse.mean()
    return(rmse)
    
#Estimators
estimators = [400,450,500,600,700,800,900,1000]
xgb_learners_cv_rmse = [rmse_cv(xgb.XGBRegressor(n_estimators=n,max_depth=3,learning_rate=0.1)) for n in estimators]

#Learning rate
etas = [0.001,0.005,0.01,0.05,0.1, 0.2, 0.3]
xgb_learnrate_cv_rmse = [rmse_cv(xgb.XGBRegressor(n_estimators=700,max_depth=3, learning_rate=eta)) for eta in etas]

#Max-depth
depths = [1,2,3,4,5]
xgb_depth_cv_rmse = [rmse_cv(xgb.XGBRegressor(n_estimators=700,max_depth=depth)) for depth in depths]



def rmse_xgb_cv(x):
    rmse = (-cross_val_score(xgb.XGBRegressor(n_estimators=int(x[0]),max_depth=int(x[1]),learning_rate=abs(x[2])), X_train, y, scoring='neg_mean_squared_error', cv=5))**0.5
    rmse = rmse.mean()
    return(rmse)

from scipy.optimize import minimize

tuned = minimize(rmse_xgb_cv,[200,3,0.1], options={'disp':True, 'ftol':0.1,'maxiter':1,'maxfev':1})

    
model_xgb = xgb.XGBRegressor(n_estimators=1000, max_depth=2, learning_rate=0.1, reg_lambda = 15, early_stopping_rounds=5)

model_xgb.fit(X_train,y)
y_hat = model_xgb.predict(X_train)
xgb_star_tr_rmse = np.mean((y-y_hat)**2)**0.5

xgb_preds = np.expm1(model_xgb.predict(X_test))


#Submission csv
submission = pd.read_csv('sample_submission.csv')
submission['SalePrice'] = xgb_preds
submission.to_csv('submission.csv',index=False)



