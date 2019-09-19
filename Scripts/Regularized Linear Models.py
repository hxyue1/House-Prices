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
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, Lasso, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from scipy.optimize import minimize_scalar

def rmse_lasso(lambda2):
    rmse = (-cross_val_score(Lasso(alpha=lambda2), X_train, y, scoring='neg_mean_squared_error', cv=5))**0.5
    rmse = rmse.mean()
    return(rmse)

#cv_lasso = [rmse_cv(Lasso(alpha=lambda2)).mean() for lambda2 in lambdas2]
lambda2_solve = minimize_scalar(rmse_lasso,bounds=(0,10))
lambda2_star = lambda2_solve['x']

#Estimating lasso with 'optimal' lambda
lasso_star = Lasso(alpha=lambda2_star).fit(X_train,y)
lasso_coef = pd.Series(lasso_star.coef_,index=X_train.columns)
cv_lasso_star = rmse_cv(Lasso(alpha=lambda2_star))
y_hat_lasso = np.matmul(X_train,lasso_coef) + lasso_star.intercept_

lasso_star_tr_rmse = (np.mean((y-y_hat_lasso)**2))**0.5
lasso_star_tr_rmse_alt = lasso_star.predict(X_train)

y_hat_test = np.exp(lasso_star.predict(X_test))-1

#Submission csv
submission = pd.read_csv('sample_submission.csv')
submission['SalePrice'] = y_hat_test
submission.to_csv('submission.csv',index=False)



