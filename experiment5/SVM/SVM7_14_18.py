#final
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR  # Support Vector Regression
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
dataset = pd.read_csv('../data/output_14_18_7.csv', parse_dates=['Time'])

# Convert 'Time' to datetime
dataset['Time'] = pd.to_datetime(dataset['Time'], format='%m/%d/%Y %H:%M')

# Extract hour and minute
dataset['Hour'] = dataset['Time'].dt.hour
dataset['Minute'] = dataset['Time'].dt.minute

# Select the relevant features
X = dataset[['Month', 'Year', 'Hour', 'Minute', 'WeekDay', 'TimeOfDay', 'Semester']]  # Feature matrix
y = dataset['sensor_mo.mean']  # Target labels (e.g., Present=1, Absent=0)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

# Fit the Support Vector Regression (SVM)
svm = SVR(kernel='rbf')  # You can adjust the kernel and hyperparameters as needed
svm.fit(X_train, y_train)

# Generate predictions for the test set
y_pred = svm.predict(X_test)

# Plot the predicted attendance against the actual attendance
plt.scatter(y_test, y_pred)
plt.xlabel('Actual attendance')
plt.ylabel('Predicted attendance')
plt.title('SVM Regression: Attendance')
plt.show()

# Calculate RMSE (Root Mean Squared Error)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

# Calculate other metrics
meanAbErr = metrics.mean_absolute_error(y_test, y_pred)
meanSqErr = metrics.mean_squared_error(y_test, y_pred)
r_squared = svm.score(X, y)

# Print the results
print('Root Mean Square Error (RMSE):', rmse)
print('Mean Absolute Error (MAE):', meanAbErr)
print('R squared: {:.2f}'.format(r_squared * 100))
print('Mean Square Error (MSE):', meanSqErr)
