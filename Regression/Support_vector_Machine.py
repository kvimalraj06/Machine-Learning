# SVR

# Importing the libraries
import numpy as np#
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set is neglected czzz we have only less values in our datasets

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))# to get 1 row and 1 column

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')#rbf is used for non linear and also it is the default one
regressor.fit(X, y)#fitting svr to our x and y

# Predicting a new result
y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))#to transform the scaled x values and np.array is used to convert the numerical value to array bczz transform accepts only the array
y_pred = sc_y.inverse_transform(y_pred)#to transform the scaled y values

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

