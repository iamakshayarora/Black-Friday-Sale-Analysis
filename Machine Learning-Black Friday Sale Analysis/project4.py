# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 23:14:26 2019

@author: Akshay
RULE BASED LEARNING
"""
import pandas as pd
import numpy as np
from sklearn import metrics
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
import collections
dataset = pd.read_csv("F:\\6th Sem\\Machine Learning\\project\\train.csv")
test = pd.read_csv("F:\\6th Sem\\Machine Learning\\project\\test.csv")
train_df = pd.read_csv("F:\\6th Sem\\Machine Learning\\project\\train_modified.csv")
test_df = pd.read_csv("F:\\6th Sem\\Machine Learning\\project\\test_modified.csv")
target = 'Purchase'
predictors = train_df.columns.drop(['Purchase','Product_ID','User_ID'])
IDcol = ['User_ID','Product_ID']
train_df['User_ID'] = train_df['User_ID'] - 1000000
test_df['User_ID'] = test_df['User_ID'] - 1000000

enc = LabelEncoder()
train_df['User_ID'] = enc.fit_transform(train_df['User_ID'])
test_df['User_ID'] = enc.transform(test_df['User_ID'])

train_df['Product_ID'] = train_df['Product_ID'].str.replace('P00', '')
test_df['Product_ID'] = test_df['Product_ID'].str.replace('P00', '')

scaler = StandardScaler()
train_df['Product_ID'] = scaler.fit_transform(train_df['Product_ID'].values.reshape(-1, 1))
test_df['Product_ID'] = scaler.transform(test_df['Product_ID'].values.reshape(-1, 1))

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

def find_node(tree_, current_node, search_node, features):
    
    child_left = tree_.children_left[current_node]
    child_right = tree_.children_right[current_node]

    split_feature = str(features[tree_.feature[current_node]])
    split_value = str(tree_.threshold[current_node])


    if child_left != -1:
        if child_left != search_node:
            left_one = find_node(tree_, child_left, search_node, features)
        else:
            return(str(split_feature)+" <= "+str(split_value))
    else:
        return ""

    if child_right != -1:
        if child_right != search_node:
            right_one = find_node(tree_, child_right, search_node, features)
        else:
            return(str(split_feature)+" > "+str(split_value))
    else:
        return ""


    if len(left_one)>0:
        return(str(split_feature)+" <= "+str(split_value)+", "+left_one)
    elif len(right_one)>0:
        return(str(split_feature)+" > "+str(split_value)+","+right_one)
    else:
        return ""
    
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


print('\nRule Based Learning')

#print(train_df)
'''newtrain = train_df.applymap(encode_units)
print(newtrain)
newtrain=newtrain.dropna()
predictors1 = train_df.columns.drop(['Product_ID','User_ID','Marital_Status','Stay_In_Current_City_Years'])
frequent_itemsets = apriori(newtrain[predictors1], min_support=0.07, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print(rules)
print(rules[ (rules['lift'] > 1.0) & (rules['confidence'] > 0.73)]) 
'''
X=train_df[predictors].loc[:2000,]
y=train_df[target].loc[:2000,]
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X,y)
print(clf)

pd.DataFrame(clf.decision_path(X).toarray()).head(5)
pd.concat([X.reset_index(drop=True),pd.DataFrame(clf.decision_path(X).toarray())],1).head(5)

print("\n\n",find_node(tree_ = clf.tree_, current_node = 0, search_node = 13, features = X.columns.tolist()),"\n\n")
print(dataset[(dataset['Purchase'] >= 10000)])

dTree3 = DecisionTreeRegressor(max_depth = 6)
commonfit(dTree3, train_df, test_df, predictors, target, IDcol, 'DT-new.csv')

Xrules = pd.concat([X.reset_index(drop=True),pd.DataFrame(dTree3.decision_path(X).toarray()).iloc[:,1:]],1)


tree_model = DecisionTreeRegressor()
tree_model.fit(Xrules, y)\

commonfit(tree_model, train_df, test_df, predictors, target, IDcol, 'DT-new.csv')