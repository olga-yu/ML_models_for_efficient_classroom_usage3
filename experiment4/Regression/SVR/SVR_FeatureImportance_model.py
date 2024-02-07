import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/wiut-tutor2022/PHD_attendance_prediction/master/Data/output_lecture_seminar_processed3.csv', parse_dates=['date'])

# Select the relevant features
feature_cols = ['week', 'day', 'time_of_day', 'class_type', 'faculty', 'school', 'joint', 'status', 'degree', 'enrollment', 'class_duration']

# Extract the features and target variable
X = dataset[feature_cols]
y = dataset['normalized_attendance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

# Fit the SVR model
svr = SVR(kernel='linear')
svr.fit(X_train, y_train)

# Generate predictions for the test set
y_pred = svr.predict(X_test)

# Compute the evaluation metrics
meanAbErr = metrics.mean_absolute_error(y_test, y_pred)
meanSqErr = metrics.mean_squared_error(y_test, y_pred)
rootMeanSqErr = np.sqrt(meanSqErr)
r_squared = svr.score(X, y)

print('Root Mean Square Error:', rootMeanSqErr)
print('Mean Absolute Error:', meanAbErr)
print('R squared: {:.2f}'.format(r_squared * 100))
print('Mean Square Error:', meanSqErr)

# Plot the feature importance
coef = pd.Series(svr.coef_[0], index=X.columns)
importance = np.abs(coef)
importance = importance.sort_values(ascending=False)
plt.bar(importance.index, importance)
plt.xticks(rotation=25)
plt.xlabel('Features')
plt.ylabel('Coefficient')
plt.title('Feature Importance for Support Vector MLR')
plt.show()