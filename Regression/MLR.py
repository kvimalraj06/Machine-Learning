#data preproceesing
#importing the libraries
import numpy as np #necessary libraries for machine learning models
import pandas as pd
import matplotlib.pyplot as mp




#import the datasets
datasets = pd.read_csv('50_startups.csv')#to import the datasets
X = datasets.iloc[:, :-1].values #independent variables
Y = datasets.iloc[:, 4].values#dependent variables

#encode the cataogorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder#to convert catogorical data to encoded values
labelencoder_X = LabelEncoder()
X[:, 3] =labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])#dummyencoding
X = onehotencoder.fit_transform(X).toarray()

#to avoid dummy variable trap
X = X[:, 1:]

#splitting the datsets into training and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test =train_test_split(X, Y, test_size = 0.2, random_state = 0)#to split train and test sets for both independent and dependent variables

#Multiple Linear Regression model
#fitting the Multiple regression model into our training set
from sklearn.linear_model import LinearRegression
reg = LinearRegression()#creating object for class 
reg.fit(X_train, Y_train)#to fit the Multiple regression line for trained set values

#predicting the test set values
Y_pred = reg.predict(X_test)#predicted the y_test values based on X_test
