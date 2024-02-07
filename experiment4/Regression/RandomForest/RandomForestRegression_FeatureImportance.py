import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
dataset = pd.read_csv(r'../../Data/output_lecture_seminar_processed3.csv', parse_dates=['date'])

# Select the relevant features
feature_cols = ['week', 'day', 'time_of_day', 'class_type', 'faculty', 'school', 'joint', 'status', 'degree', 'enrollment', 'class_duration']

# Extract the features and target variable
X = dataset[feature_cols]
y = dataset['normalized_attendance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = None)

# Fit the random forest regression model
regr = RandomForestRegressor(n_estimators=100, max_depth=5)
regr.fit(X_train, y_train)

# Generate predictions for the test set
y_pred = regr.predict(X_test)

# Plot the predicted attendance against the actual attendance
plt.scatter(y_test, y_pred)
plt.xlabel('Actual attendance')
plt.ylabel('Predicted attendance')
plt.title('Random Forest MLR: 10 features')
plt.show()

# Compute the evaluation metrics

meanAbErr = metrics.mean_absolute_error(y_test, y_pred)
meanSqErr = metrics.mean_squared_error(y_test, y_pred)
rootMeanSqErr = np.sqrt(meanSqErr)
r_squared = regr.score(X, y)


print('Root Mean Square Error:', rootMeanSqErr)
print('Mean Absolute Error:', meanAbErr)
print('R squared: {:.2f}'.format(r_squared * 100))
print('Mean Square Error:', meanSqErr)

importances = regr.feature_importances_
std = np.std([tree.feature_importances_ for tree in regr.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

plt.figure()
plt.title("Feature importance for Random Forest MLR")
plt.bar(range(X.shape[1]), importances[indices], yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=25)
plt.xlim([-1, X.shape[1]])
plt.show()

