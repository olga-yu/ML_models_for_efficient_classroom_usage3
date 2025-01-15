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
dataset_train = pd.read_csv('../data/output_9_18.csv', header=0, index_col=0)
training_set = dataset_train.iloc[:, 0:1].values

# Test stationarity
print("Testing stationarity for raw training data...")
test_stationarity(training_set)

# Apply differencing if non-stationary
training_set_diff = np.diff(training_set, axis=0)
print("Testing stationarity for differenced data...")
test_stationarity(training_set_diff)

# Use differenced data for training
training_set_processed = training_set_diff

# Normalize the data
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set_processed)

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
real_attendance = dataset_test.iloc[-testing_set_size:, 0:1].values

# Combine datasets for scaling
inputs = np.vstack((training_set[-testing_set_size:], real_attendance))
inputs_scaled = sc.transform(inputs)

# Prepare testing data
X_test = []
for i in range(len(inputs_scaled) - testing_set_size, len(inputs_scaled)):
    X_test.append(inputs_scaled[i:i+1, 0])
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
