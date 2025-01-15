import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller

def calculate_mape(y_true, y_pred):
    # Avoid division by zero by adding a small constant to y_true
    percentage_errors = np.abs((y_true - y_pred) / (y_true + 1e-8))
    mape = np.mean(percentage_errors) * 100
    return mape

def test_stationarity(data):
    result = adfuller(data.flatten())
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] > 0.05:
        print("Data is non-stationary. Consider differencing.")
    else:
        print("Data is stationary.")

# Parameters
testing_set_size = 19

# Load and preprocess the dataset
# dataset_train = pd.read_csv('../data/output_9_18.csv', header=0, index_col=0)
#
# # Extract "Extracted_Time" column and ensure it's in datetime format
# dataset_train['Extracted_Time'] = pd.to_datetime(dataset_train['Extracted_Time'], format='%m/%d/%Y %H:%M')

dataset_train = pd.read_csv('../data/output_9_18.csv', header=0, index_col=0)

# Try converting 'Extracted_Time' to datetime, letting pandas infer the format
dataset_train['Extracted_Time'] = pd.to_datetime(dataset_train['Extracted_Time'], errors='coerce', dayfirst=True)
dataset_train['Extracted_Time'] = pd.to_datetime(dataset_train['Extracted_Time'].apply(lambda x: '2024-01-15 ' + str(x)), format='%Y-%m-%d %H:%M')

# Check if any NaT (Not a Time) entries exist after conversion
print(dataset_train['Extracted_Time'].isna().sum())




# Use Extracted_Time as index
dataset_train.set_index('Extracted_Time', inplace=True)

# Assuming the data column of interest is 'sensor_mo.mean'
training_set = dataset_train['sensor_mo.mean'].values

# Test stationarity
print("Testing stationarity for raw training data...")
test_stationarity(training_set)

dataset_test['Extracted_Time'] = pd.to_datetime(dataset_test['Extracted_Time'].apply(lambda x: '2024-01-15 ' + str(x)), format='%Y-%m-%d %H:%M')

# Apply differencing if non-stationary
training_set_diff = np.diff(training_set, axis=0)
print("Testing stationarity for differenced data...")
test_stationarity(training_set_diff)

# Use differenced data for training
training_set_processed = training_set_diff

# Normalize the data
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set_processed.reshape(-1, 1))

# Prepare training data
X_train = []
y_train = []
for i in range(0, (len(training_set_scaled) - testing_set_size)):
    X_train.append(training_set_scaled[i:i+1, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Configure ARIMAX model
order = (2, 1, 1)  # Adjusted d=1 to account for differencing
exog = X_train

# Fit ARIMAX model
model = ARIMA(endog=y_train, exog=exog, order=order)
arimax_pred = model.fit()
print(arimax_pred.summary())

# Load and preprocess the testing dataset
dataset_test = pd.read_csv('../data/output_9_18.csv', header=0, index_col=0)

# Extract "Extracted_Time" for testing dataset and ensure it's in datetime format
dataset_test['Extracted_Time'] = pd.to_datetime(dataset_test['Extracted_Time'], format='%m/%d/%Y %H:%M')

# Use Extracted_Time as index
dataset_test.set_index('Extracted_Time', inplace=True)

# Assuming the same column for sensor motion data in the test set
real_attendance = dataset_test['sensor_mo.mean'].iloc[-testing_set_size:].values

# Combine datasets for scaling
inputs = np.vstack((training_set_scaled[-testing_set_size:], sc.transform(real_attendance.reshape(-1, 1))))

# Prepare testing data
X_test = []
for i in range(len(inputs) - testing_set_size, len(inputs)):
    X_test.append(inputs[i:i+1, 0])
X_test = np.array(X_test)

# Make predictions
predicted_presence = arimax_pred.predict(start=len(y_train), end=len(y_train) + testing_set_size - 1, exog=X_test)

# Rescale predictions to original scale
predicted_presence = sc.inverse_transform(predicted_presence.reshape(-1, 1))

# Calculate MAPE
mape = calculate_mape(real_attendance, predicted_presence)
print(f'Single Motion output 9:00 to 18:00 ARIMAX MAPE: {mape:.3f}%')

# Plot results
plt.plot(real_attendance, color='red', label='Real Motion Data')
plt.plot(predicted_presence, color='blue', label='Predicted Motion Data')
plt.title('ARIMAX Prediction of Sensor Motion Data')
plt.xlabel('Time')
plt.ylabel('Sensor Motion Mean')
plt.legend()
plt.show()
