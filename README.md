# house-prices

Repository containing all files my working files to do with the Kaggle House Prices competition. This competition uses the Ames Housing dataset and is publically available. It was compiled by Dean De Cock for education purposes. While I wish I could link to the original dataset, the link in De Cock's paper is broken, so I've included the data set in my repository to make it easier to run. You can also access the data set through Kaggle as well. This is a relatively small project that I finished the bulk of in about a week. In future I may open it up, to reinvestigate hyperparamter tuning protocols.

## Metrics

Kaggle uses rmse of the difference between the predicted log sale price and the actual sale price for a hidden test set as the evaluation metric. My notebook ML-models will proxy that using five-fold cross-validation. 

# Notebooks

As part of my workflow, I've included two main notebooks which cover data cleaning and exploration (EDA.ipynb), and training and evaluation of machine learning models (ML-models.ipynb).

## Packages used

I used Pandas, Numpy and Seaborn for initial data exploration. Sklearn was used heavily in the ML notebook, and xgboost for the gradient boosting model. Models were tuned using bayesian optimisation from the bayesian-optimization library.

# Findings

I investigated four main models, Ridge Regression, Lasso Regression, Random Forests and Gradient Boosting (implemented in xgboost). Ridge, Lasso and the xgboost implementation had very similar performance. On the leaderboard however, the Lasso performed the best. Even though I tried blending the predictions together, I was still not able to beat the simple Lasso. My best score was 0.12357, placing me at rank 1419 out of 4635 (27/9/19) and in the top 30%.
