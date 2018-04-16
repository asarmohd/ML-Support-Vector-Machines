# SVR
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='poly')
regressor.fit(x, y)

# Predicting a new result
y_pred = regressor.predict(1)

# Visualising the SVR results
plt.scatter(x, y, color = 'red')
plt.plot(x,regressor.predict(x), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

