# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 14:51:09 2020

@author: Poorvahab
"""
import pandas as pd

dataset = pd.read_csv('vehicle_csv.csv')
print('\n________ Dataset _________')
print(dataset)

print('__________Label Names ________')
label_names = dataset["Class"]
print(label_names)
#print('__________Label ________')
#labels = dataset["target"]
#print(labels)
print('__________Feature_names ________')
feature_names = dataset["feature_names"]
print(feature_names) 
    
x = dataset.iloc[:,:-1].values
print('______________ x is __________________')
print(x)
y = dataset.iloc[:,1].values
print('______________ y is __________________')
print(y)
print('\n_________ Training __________')
from sklearn.model_selection import train_test_split
xtrain, xtest,ytrain,ytest = train_test_split(x,y,test_size=1 / 3)
print('\n_________ Modeling __________')
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtrain,ytrain)
ypred = regressor.predict(xtest)
print('_____ xTest ________')
print(xtest)
print('_____ xPredict ________')
print('preict')
print(ypred)
