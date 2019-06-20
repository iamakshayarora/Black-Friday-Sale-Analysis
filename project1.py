# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 18:20:55 2019

@author: Akshay
ANALYSING
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


pd.options.display.max_columns = 200
dataset = pd.read_csv("F:\\6th Sem\\Machine Learning\\project\\train.csv")
test = pd.read_csv("F:\\6th Sem\\Machine Learning\\project\\test.csv")
#submission = pd.read_csv("F:\\6th Sem\\Machine Learning\\project\\sub.csv")

print("\n\n",dataset.head())
print("\n\n",dataset.describe(),"\n\n")
print(dataset.info())

idsUnique = len(set(dataset.User_ID))
idsTotal = dataset.shape[0]
idsDupli = idsTotal - idsUnique
print("\n\nThere are " + str(idsDupli) + " duplicate IDs for " + str(idsTotal) + " total entries")

print ("\n\nSkew is:", dataset.Purchase.skew())
print("Kurtosis: %f" % dataset.Purchase.kurt())

numeric_features = dataset.select_dtypes(include=[np.number])

sns.countplot(dataset.Occupation)
plt.show()
sns.countplot(dataset.Marital_Status)
plt.show()
sns.countplot(dataset.Product_Category_1)
plt.show()
sns.countplot(dataset.Product_Category_2)
plt.show()
sns.countplot(dataset.Product_Category_3)
plt.show()
sns.countplot(dataset.Gender)
plt.show()
sns.countplot(dataset.Stay_In_Current_City_Years)
plt.show()
sns.countplot(dataset.City_Category)
plt.show()

corr = numeric_features.corr()
print ("\n\nCorrelation from Purchase\n",corr['Purchase'].sort_values(ascending=False),"\n")

print("Correlation Matrix")
f, ax = plt.subplots(figsize=(9, 5))
sns.heatmap(corr, vmax=.8,annot_kws={'size': 14}, annot=True);


Occupation_pivot = dataset.pivot_table(index='Occupation', values="Purchase", aggfunc=np.mean)

Occupation_pivot.plot(kind='bar', color='darkorange',figsize=(9,5))
plt.xlabel("Occupation")
plt.ylabel("Purchase")
plt.title("Occupation vs Purchase")
plt.show()

Product_Category_1_pivot=dataset.pivot_table(index='Product_Category_1', values="Purchase", aggfunc=np.mean)

Product_Category_1_pivot.plot(kind='bar', color='darkorange',figsize=(9,5))
plt.xlabel("Product_1")
plt.ylabel("Purchase")
plt.title("Product_1 vs Purchase")
plt.show()

roduct_Category_2_pivot=dataset.pivot_table(index='Product_Category_2', values="Purchase")

roduct_Category_2_pivot.plot(kind='bar', color='darkgreen',figsize=(9,5))
plt.xlabel("Product_2")
plt.ylabel("Purchase")
plt.title("Product_2 vs Purchase")
plt.show()

Age1= dataset.pivot_table(index='Age', values="Purchase", aggfunc=np.mean)
Age1.plot(kind='bar', color='darkgreen',figsize=(9,5))
plt.xlabel("Age")
plt.ylabel("Purchase")
plt.title("Age vs Purchase")
plt.show()

Occupation1 = dataset.pivot_table(index='Marital_Status', values="Purchase", aggfunc=np.mean)

Occupation1.plot(kind='bar', color='darkgreen',figsize=(9,5))
plt.xlabel("Marital_Status")
plt.ylabel("Purchase")
plt.title("Marital_Status vs Purchase")
plt.show()

City1 = dataset.pivot_table(index='City_Category', values="Purchase", aggfunc=np.mean)

City1.plot(kind='bar', color='darkgreen',figsize=(9,5))
plt.xlabel("City_Category")
plt.ylabel("Purchase")
plt.title("City_Category vs Purchase")
plt.show()