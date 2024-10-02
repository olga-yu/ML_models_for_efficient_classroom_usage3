import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


# Load the dataset
#dataset = pd.read_csv(r'../../Data/output_lecture_seminar_processed3.csv', parse_dates=['date'])
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


# Plot the predicted attendance against the actual attendance
plt.scatter(y_test, y_pred)
plt.xlabel('Actual attendance')
plt.ylabel('Predicted attendance')
plt.title('Decision Tree MLR: 10 features')
plt.show()

meanAbErr = metrics.mean_absolute_error(y_test, y_pred)
meanSqErr = metrics.mean_squared_error(y_test, y_pred)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r_squared = regr.score(X, y)

print('Root Mean Square Error:', rootMeanSqErr)
print('Mean Absolute Error:', meanAbErr)
print('R squared: {:.2f}'.format(r_squared * 100))
print('Mean Square Error:', meanSqErr)

# Plot the Residuals
plt.scatter(y_pred, y_test - y_pred)
plt.xlabel('Predicted attendance')
plt.ylabel('Residuals')
plt.title('Residual plot')
plt.show()