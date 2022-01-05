import matplotlib.pyplot as  plt 
import pandas as pd 
import numpy as np
from pandas.core.frame import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math

#Load the data set
housing = pd.read_csv("data.csv.csv")

#Analysis of data
x = housing.iloc[:,:-1]
y = housing.iloc[:,-1]


#Splitting of train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


#Now applying RandomForestRegressor and evaluation
model = RandomForestRegressor()
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)


#Checking the MSE and RMSE
mse = mean_squared_error(y_test, y_predicted)
print("Mean squared error is:", mse)
rmse = math.sqrt(mse)
print("root mean square :", rmse)

#plotting the graph of predicted and actual values

plt.plot(y_predicted,y_test, 'ro' )
plt.show()


