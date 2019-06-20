# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 21:20:26 2019

@author: Akshay
PREDICTING
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

pd.options.display.max_columns = 200
train_df = pd.read_csv("F:\\6th Sem\\Machine Learning\\project\\train_modified.csv")
test_df = pd.read_csv("F:\\6th Sem\\Machine Learning\\project\\test_modified.csv")

target = 'Purchase'
IDcol = ['User_ID','Product_ID']

def commonfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    alg.fit(dtrain[predictors], dtrain[target])
        
    dtrain_predictions = alg.predict(dtrain[predictors])

    cv_score = cross_val_score(alg, dtrain[predictors],(dtrain[target]) , cv=20, scoring='neg_mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    print("\nModel Report")
    print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error((dtrain[target]).values, dtrain_predictions)))
    print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
    dtest[target] = alg.predict(dtest[predictors])
    
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=False)


LR = LinearRegression(normalize=True)
print("\nLinear Regression")
predictors = train_df.columns.drop(['Purchase','Product_ID','User_ID'])
commonfit(LR, train_df, test_df, predictors, target, IDcol, 'LR.csv')
coef1 = pd.Series(LR.coef_, predictors).sort_values()
coef1.plot(kind='bar', title='Model Coefficients')
plt.show()


print("\nRidge Regression")
RR1 = Ridge(alpha=0.05,normalize=True)
commonfit(RR1, train_df, test_df, predictors, target, IDcol, 'RR.csv')
coef1 = pd.Series(RR1.coef_, predictors).sort_values()
coef1.plot(kind='bar', title='Model Coefficients')
plt.show()



print("\nDecision Tree Regression")
DT = DecisionTreeRegressor(max_depth=15, min_samples_leaf=200)
commonfit(DT, train_df, test_df, predictors, target, IDcol, 'DT.csv')
importances = DT.feature_importances_
indices = np.argsort(importances)[::-1]
print("\nFeature ranking:")
for f in range(train_df[predictors].shape[1]):
    print("x%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
X=train_df[predictors]
plt.figure()
plt.title("\nFeature importances")
plt.bar(range(X.shape[1]), importances[indices],color="y", align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

print("\nXGB Regression")
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
print(my_model.fit(train_df[predictors], train_df[target], early_stopping_rounds=5, eval_set=[(test_df[predictors], test_df[target])], verbose=False))

train_df_predictions = my_model.predict(train_df[predictors])
predictions = my_model.predict(test_df[predictors])
print("\nMean Absolute Error : " + str(mean_absolute_error(predictions, test_df[target])))
print("\nRMSE : %.4g" % np.sqrt(metrics.mean_squared_error((train_df[target]).values, train_df_predictions)))
IDcol.append(target)
submission = pd.DataFrame({ x: test_df[x] for x in IDcol})
submission.to_csv("XGB.csv", index=False)

importances = my_model.feature_importances_
indices = np.argsort(importances)[::-1]
print("\nFeature order:")
for f in range(train_df[predictors].shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

print("\nRandom Forest Regression")
RF=RandomForestRegressor(n_estimators=200, max_depth=3)
print(RF)
commonfit(RF, train_df, test_df, predictors, target, IDcol, 'RF.csv')
RF.fit(train_df[predictors], train_df[target])
train_df_predictions = RF.predict(train_df[predictors])
predictions = RF.predict(test_df[predictors])

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_df[target])))
print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error((train_df[target]).values, train_df_predictions)))

importances = RF.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature order:")
for f in range(train_df[predictors].shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

X=train_df[predictors]
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],color="y", align="center")
plt.xlim([-1, X.shape[1]])
plt.show()