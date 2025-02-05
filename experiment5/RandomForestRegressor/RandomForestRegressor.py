import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from imblearn.over_sampling import SMOTENC  # SMOTE for categorical features

# Load dataset
dataset = pd.read_csv('../data/augmented_dataset.csv', parse_dates=['Time'])

# Convert 'Time' to datetime
dataset['Time'] = pd.to_datetime(dataset['Time'], format='%m/%d/%Y %H:%M')

# Extract hour and minute
dataset['Hour'] = dataset['Time'].dt.hour
dataset['Minute'] = dataset['Time'].dt.minute

# Select features and target variable
X = dataset[['Month', 'Year', 'Hour', 'Minute', 'WeekDay', 'TimeOfDay', 'Semester']]
y = dataset['sensor_mo.mean']  # Target variable

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply SMOTE to balance the training data
smote = SMOTENC(categorical_features=[0, 1, 4, 5, 6], random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_resampled, y_train_resampled)

# Generate predictions
y_pred = rf.predict(X_test)

# Plot actual vs predicted values
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Actual Attendance')
plt.ylabel('Predicted Attendance')
plt.title('Random Forest Regression: Attendance, 9:00 - 18:00')
plt.show()

# Evaluate model performance
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
r_squared = rf.score(X_test, y_test)

# Print evaluation metrics
print('Root Mean Square Error (RMSE):', rmse)
print('Mean Absolute Error (MAE):', mae)
print('Mean Square Error (MSE):', mse)
print('R Squared (R²): {:.2f}'.format(r_squared * 100))
