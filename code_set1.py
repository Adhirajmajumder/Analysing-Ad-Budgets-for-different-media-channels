# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 20:52:31 2020

@author: ADHIRAJ MAJUMDAR
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("./dataset/Advertising Budget and Sales.csv",index_col=None)
print(data.head())
del data["Unnamed: 0"]
#print(data.columns)
X_feature = data[["TV Ad Budget ($)","Radio Ad Budget ($)","Newspaper Ad Budget ($)"]]
Y_target = data["Sales ($)"]
#Split the dataset (by default, 75% is the training data and 25% is the testing data)
X_train, X_test, Y_train, Y_test = train_test_split(X_feature,Y_target,test_size=0.25,random_state=1)
#Create a linear regression model
lineReg = LinearRegression()
lineReg.fit(X_train,Y_train)
#Print the intercept and coefficients 
print(lineReg.intercept_)
print(lineReg.coef_)
#Predict the outcome for the testing dataset
y_pred = lineReg.predict(X_test)
y_pred
#Import required libraries for calculating MSE (mean square error)
from sklearn import metrics
#Calculate the MSE
print(np.sqrt(metrics.mean_squared_error(Y_test,y_pred)))
