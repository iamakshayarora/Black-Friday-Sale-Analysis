# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 19:35:13 2019

@author: Akshay
CLEANING OF DATASET
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


pd.options.display.max_columns = 200
dataset = pd.read_csv("F:\\6th Sem\\Machine Learning\\project\\train.csv")
test = pd.read_csv("F:\\6th Sem\\Machine Learning\\project\\test.csv")
#submission = pd.read_csv("F:\\6th Sem\\Machine Learning\\project\\sub.csv")

dataset['source']='train'
test['source']='test'

data = pd.concat([dataset,test], ignore_index = True, sort = False)
print(dataset.shape, test.shape, data.shape)

print("\n\nNull Value Average\n",data.isnull().sum()/data.shape[0]*100);

data["Product_Category_2"]=data["Product_Category_2"].fillna(-1.0).astype("float")
print("\n\n",data.Product_Category_2.value_counts().sort_index())
data["Product_Category_3"]=data["Product_Category_3"].fillna(-1.0).astype("float")
print("\n\n",data.Product_Category_3.value_counts().sort_index())

category_cols = data.select_dtypes(include=['object'])
for col in category_cols:
 frequency = data[col].value_counts()
 print("\n\nThis is the frequency distribution for " + col + ":")
 print(frequency)
 
 
data['Gender'],ages = pd.factorize(data['Gender'])
print("\n\n",ages)
print(data['Gender'].unique())
print(data["Gender"].value_counts())
data['Age'],ages = pd.factorize(data['Age'])
print("\n\n",ages)
print(data['Age'].unique())
print(data["Age"].value_counts())
data['Stay_In_Current_City_Years'],scc = pd.factorize(data['Stay_In_Current_City_Years'])
print("\n\n",scc)
print(data['Stay_In_Current_City_Years'].unique())
print(data['Stay_In_Current_City_Years'].value_counts())
data['City_Category'],cc = pd.factorize(data['City_Category'])
print("\n\n",cc)
print(data['City_Category'].unique())
print(data['City_Category'].value_counts())
print("\n\n")

train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]
test.drop(['source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)

train.to_csv("F:\\6th Sem\\Machine Learning\\project\\train_modified.csv",index=False)
test.to_csv("F:\\6th Sem\\Machine Learning\\project\\test_modified.csv",index=False)