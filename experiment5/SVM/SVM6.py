import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Support Vector Classification
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
dataset = pd.read_csv('../data/output_9_18.csv', parse_dates=['Time'])

# Convert 'Time' to datetime
dataset['Time'] = pd.to_datetime(dataset['Time'], format='%m/%d/%Y %H:%M')

# Extract hour and minute
dataset['Hour'] = dataset['Time'].dt.hour
dataset['Minute'] = dataset['Time'].dt.minute

# Select relevant features
X = dataset[['Month', 'Year', 'Hour', 'Minute', 'WeekDay', 'TimeOfDay', 'Semester']]  # Feature matrix
y = dataset['sensor_mo.mean']  # Target labels (e.g., Present=1, Absent=0)

# Ensure y is categorical (classification problem)
y = y.astype('int')  # Convert target to integers (0 and 1)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the Support Vector Classification model
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)

# Generate predictions for the test set
y_pred = svm.predict(X_test)

# Evaluate the model
print("\nAccuracy:", metrics.accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(metrics.classification_report(y_test, y_pred))

# Plot the actual vs. predicted labels (for visualization)
plt.scatter(y_test, y_pred)
plt.xlabel('Actual attendance')
plt.ylabel('Predicted attendance')
plt.title('SVM Classification: Attendance')
plt.show()
