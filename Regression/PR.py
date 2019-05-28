#polynomial regression
#data preprocessing template
#importing the libraries
import numpy as np #necessary libraries for machine learning models
import pandas as pd
import matplotlib.pyplot as mp

#import the datasets
datasets = pd.read_csv('position_Salaries.csv')#to import the datasets
X = datasets.iloc[:, 1:2].values #independent variablesans also to consider x as matrix
Y = datasets.iloc[:, 2].values#dependent variables

# Because of less availability of data split is neglected
"""#splitting the datsets into training and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test =train_test_split(X, Y, test_size = 1/3, random_state = 0)#to split train and test sets for both independent and dependent variables"""

#simple Linear Regression model for to compare with polynomial Regresssion model
#fitting the linear regression model to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()#creating object for class 
lin_reg.fit(X, Y)#to fit the linear regression line for X and Y

#fitting the polynomial regression model to the dataset
from sklearn.preprocessing import PolynomialFeatures
pol_reg = PolynomialFeatures(degree = 4)#creating object
X_pol = pol_reg.fit_transform(X)#to add polynomial terms for the matrix of features X 
lin_reg_2 = LinearRegression()#create separate object for polynomial regressiom
lin_reg_2.fit(X_pol, Y)#fitting with X_pol and and Y(polynoial regression)

#visualizing our Linear regression model
mp.scatter(X, Y, color = "black")# to plot values
mp.plot(X,  lin_reg.predict(X), color = "red")#to draw the regression line
mp.show()#to display the graph

#visualizing our polynomial regression model
X_grid = np.arange(min(X), max(X), 0.1)#for more accuracy
X_grid = X_grid.reshape(len(X_grid), 1)
mp.scatter(X, Y, color = "black")# to plot values
mp.plot(X,  lin_reg_2.predict( pol_reg.fit_transform(X)), color = "red")#to draw the regression line
mp.show()#to display the graph


 lin_reg.predict([[6.5]])
lin_reg_2.predict( pol_reg.fit_transform([[6.5]]))

