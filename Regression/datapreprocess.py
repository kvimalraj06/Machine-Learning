#data preproceesing
#importing the libraries
import numpy as np #necessary libraries for machine learning models
import pandas as pd
import matplotlib.pyplot as mp

#import the datasets
datasets = pd.read_csv('Data.csv')#to import the datasets
X = datasets.iloc[:, :-1].values #independent variables
Y = datasets.iloc[:, 3].values#dependent variables

#To take care of the missing values
from sklearn.impute import SimpleImputer#library and class for missing data
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')#to replace missing data with mean
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#encode the cataogorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder#to convert catogorical data to encoded values
labelencoder_X = LabelEncoder()
X[:, 0] =labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])#dummyencoding
X = onehotencoder.fit_transform(X).toarray()
