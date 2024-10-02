import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/wiut-tutor2022/PHD_attendance_prediction/master/Data/output_lecture_seminar_processed3.csv', parse_dates=['date'])

# Select the relevant features
feature_cols = ['week', 'day', 'time_of_day', 'class_type', 'faculty', 'school', 'joint', 'status', 'degree', 'enrollment', 'class_duration']

# Extract the features and target variable
X = dataset[feature_cols]
y = dataset['normalized_attendance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

# Fit the decision tree regression model
regr = DecisionTreeRegressor(max_depth=5)
regr.fit(X_train, y_train)

# Generate predictions for the test set
y_pred = regr.predict(X_test)

# Plot the residuals
plt.scatter(y_pred, y_test - y_pred)
plt.title('Residual plot')
plt.xlabel('Predicted attendance')
plt.ylabel('Residuals')
plt.show()

# Plot the feature importances
importances = regr.feature_importances_
indices = np.argsort(importances)[::-1]
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=25)
plt.title('Feature importance for Decision Tree MLR')
plt.show()

meanAbErr = metrics.mean_absolute_error(y_test, y_pred)
meanSqErr = metrics.mean_squared_error(y_test, y_pred)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r_squared = regr.score(X, y)

print('Root Mean Square Error:', rootMeanSqErr)
print('Mean Absolute Error:', meanAbErr)
print('R squared: {:.2f}'.format(r_squared * 100))
print('Mean Square Error:', meanSqErr)