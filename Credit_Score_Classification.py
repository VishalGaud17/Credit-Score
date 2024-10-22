#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('Credit_Score_Classification.csv')

df.head()

df.info()

df.isna().sum()

df.rename(columns = {'Marital Status':'Marital_Status'}, inplace = True)

df.rename(columns = {'Number of Children':'NumOfChildren'}, inplace = True)

df.rename(columns = {'Home Ownership':'HomeOwnership'}, inplace = True)

df.rename(columns = {'Credit Score':'CreditScore'}, inplace = True)

df=pd.get_dummies(df,columns=['Gender','Marital_Status','HomeOwnership'],drop_first=True)

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df.head()

df['Education'].value_counts()

edu={"Bachelor's Degree":2,"Master's Degree":3,"Doctorate":4,"High School Diploma":0,"Associate's Degree":1}

df['Education']=df['Education'].map(edu)

df['CreditScore'].value_counts()

cr_score={'High':2,'Average':1,"Low":0}

df['CreditScore']=df['CreditScore'].map(cr_score)

df.head()

df.tail()

sns.heatmap(df.corr(),annot=True)

df.info()

from sklearn.model_selection import train_test_split

df.columns

features=['Age', 'Income', 'Education', 'NumOfChildren',
       'Gender_Male', 'Marital_Status_Single', 'HomeOwnership_Rented']

X=df[features]
y=df['CreditScore']

X,y

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=17)

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)

train_pred=knn.predict(X_train)
test_pred=knn.predict(X_test)

from sklearn.metrics import classification_report

print('Train')
print(classification_report(y_train,train_pred))
print('Test')
print(classification_report(y_test,test_pred))

from sklearn.model_selection import cross_val_score

k_neighbors=np.arange(1,21)
k_neighbors

cv_scores={}
for i in k_neighbors:
    knn=KNeighborsClassifier(n_neighbors=i)
    scores=cross_val_score(knn,X,y,cv=10,scoring='accuracy')
    cv_scores[i]=scores.mean()

cv_scores

mse=[1-x for x in cv_scores.values()]
mse

min(mse)

k_neighbors[mse.index(min(mse))]

knn=KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train,y_train)

train_pred=knn.predict(X_train)
test_pred=knn.predict(X_test)

print('Train')
print(classification_report(y_train,train_pred))
print('Test')
print(classification_report(y_test,test_pred))

import pickle

with open('CRC.pkl', 'wb') as file:
            pickle.dump(knn, file)

