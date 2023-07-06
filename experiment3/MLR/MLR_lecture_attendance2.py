#
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics

# Load the dataset from a CSV file#
dataset = pd.read_csv('../../Data/westminster.csv')

# Select the relevant features
feature_cols = ['Week', 'Campus_L_attendance', 'Module_A_deadlines', 'Module_B_deadlines', 'Module_C_deadlines',]

# Extract the features and target variable
X = dataset[feature_cols]
y = dataset['Module_X_Lecture_attendance']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the Multiple Linear MLR model
mlr = LinearRegression()
mlr.fit(X_train, y_train)

# Prediction on the test set
y_pred_mlr = mlr.predict(X_test)

# Print model coefficients and intercept
print("Intercept:", mlr.intercept_)
print("Coefficients:")
print(list(zip(X.columns, mlr.coef_)))

# Calculate evaluation metrics
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(meanSqErr)

print('Root Mean Square Error:', rootMeanSqErr)
print('Mean Absolute Error:', meanAbErr)
print('R-squared:', mlr.score(X_test, y_test))

# Scatter plots to visualize the relationships
fig, axs = plt.subplots(2, 1, figsize=(8, 5))
axs = axs.flatten()

dependent_attrs = ['Week', 'Campus_L_attendance', ]

for i, attr in enumerate(dependent_attrs):
    axs[i].scatter(dataset[attr], dataset['Module_X_Lecture_attendance'], color='blue')
    axs[i].set_xlabel(attr)
    axs[i].set_ylabel('Module_X_Lecture_attendance')

plt.tight_layout()
plt.show()