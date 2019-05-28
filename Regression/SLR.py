#simple linear Regression
#data preprocessing template
#importing the libraries
import numpy as np #necessary libraries for machine learning models
import pandas as pd
import matplotlib.pyplot as mp

#import the datasets
datasets = pd.read_csv('Salary_Data.csv')#to import the datasets
X = datasets.iloc[:, :-1].values #independent variables
Y = datasets.iloc[:, 1].values#dependent variables

#splitting the datsets into training and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test =train_test_split(X, Y, test_size = 1/3, random_state = 0)#to split train and test sets for both independent and dependent variables

#simple Linear Regression model
#fitting the linear regression model into our training set
from sklearn.linear_model import LinearRegression
reg = LinearRegression()#creating object for class 
reg.fit(X_train, Y_train)#to fit the linear regression line for trained set values

#predicting the test set values
Y_pred = reg.predict(X_test)#predicted the y_test values based on X_test

#visualizing our trained values
mp.scatter(X_train, Y_train, color = "black")# to plot values
mp.plot(X_train,  reg.predict(X_train), color = "red")#to draw the regression line
mp.title("salary vs experience")
mp.xlabel("experience")
mp.ylabel("salary")
mp.show()#to display the graph

#visualizing our tested values
mp.scatter(X_test, Y_test, color = "blue")
mp.plot(X_train,  reg.predict(X_train), color = "yellow")#to predict the tested values based on the previous regression line
mp.title("salary vs experience")
mp.xlabel("experience")
mp.ylabel("salary")
mp.show()
