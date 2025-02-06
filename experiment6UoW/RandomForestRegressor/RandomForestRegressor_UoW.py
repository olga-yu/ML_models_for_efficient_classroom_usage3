import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from imblearn.over_sampling import SMOTE  # Removed KMeansSMOTE (not needed for binary)

# Load dataset
dataset = pd.read_csv('../data/UoWprocessed_motionData2025_3.csv', parse_dates=['Time'])

# Drop unnecessary columns safely
dataset = dataset.drop(columns=['Time', 'Date'], errors='ignore')

# Select features and target variable
X = dataset[["StudentID", "TimePeriod", 'date-year', 'date-month', 'date-day', 'Season', 'Weekday', 'Semester']]
y = dataset['sensor_mo.mean']  # Ensure this column exists

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Check class distribution before SMOTE
print("Before SMOTE:")
print(y_train.value_counts())

# Step 1: Apply SMOTE directly
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check class distribution after SMOTE
print("After SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

# Train Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_resampled, y_train_resampled)

# Predictions
y_pred = rf.predict(X_test)

# Model Evaluation
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
