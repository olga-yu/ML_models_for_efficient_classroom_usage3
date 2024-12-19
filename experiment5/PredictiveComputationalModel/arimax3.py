import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Absolute Percentage Error (or simply MAPE)
def calculate_mape(y_true, y_pred):
    percentage_errors = np.abs((y_true - y_pred) / y_true)
    mape = np.mean(percentage_errors) * 100
    return mape

# Function to convert time in "HH:MM" to minutes since midnight
def time_to_minutes(time_str):
    time_obj = pd.to_datetime(time_str, format='%H:%M')
    return time_obj.hour * 60 + time_obj.minute

# Load and preprocess the training dataset
dataset_train = pd.read_csv('../data/output_9_18.csv', header=0, index_col=0)

# Assuming the first column has date-time information like '2/20/2024 10:00'
dataset_train['datetime'] = pd.to_datetime(dataset_train.iloc[:, 0], format='%m/%d/%Y %H:%M')

# Extract time-related features like minutes since midnight
dataset_train['Time_in_minutes'] = dataset_train['datetime'].dt.hour * 60 + dataset_train['datetime'].dt.minute

# Extract the features and labels (Assume the attendance is in the 2nd column, adjust accordingly)
training_set = dataset_train.iloc[3:, 1:].values  # Starting from index 3 for training (adjust accordingly)
print(training_set)

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []

testing_set_size = 3

for i in range(0, (len(training_set) - testing_set_size)):
    X_train.append(training_set_scaled[i:i+1, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# ARIMAX model configuration
order = (1, 1, 1)  # ARIMA order: Autoregressive, Differencing, Moving Average orders
exog = X_train  # exogenous variables

model = ARIMA(endog=y_train, exog=exog, order=order)
arimax_pred = model.fit()

# Load and preprocess the testing dataset (last 12 months)
dataset_test = pd.read_csv('../data/output_9_18.csv', header=0, index_col=0)

# Process the datetime column in the test dataset as well
dataset_test['datetime'] = pd.to_datetime(dataset_test.iloc[:, 0], format='%m/%d/%Y %H:%M')
dataset_test['Time_in_minutes'] = dataset_test['datetime'].dt.hour * 60 + dataset_test['datetime'].dt.minute

attendance = dataset_test.iloc[-testing_set_size:, 0:1].values

dataset_total = pd.concat((dataset_train['westminster_output_9_18'], dataset_test['westminster_output_9_18']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test):].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []
for i in range((len(training_set) - testing_set_size), len(training_set)):
    X_test.append(inputs[i:i+1, 0])

X_test = np.array(X_test)

# Make predictions with ARIMAX model for the last 12 months
predicted_attendance = arimax_pred.predict(start=len(y_train), end=len(y_train) + testing_set_size - 1, exog=X_test)

# Rescale the predicted attendance
predicted_attendance = sc.inverse_transform(predicted_attendance.reshape(-1, 1))

# Calculate MAPE
mape = calculate_mape(attendance, predicted_attendance)
print(f'Single Attendance ARIMAX MAPE: {mape:.3f}%')

# Find p, d, q values of ARIMA
result = adfuller(dataset_train['Module_X_Lecture_attendance'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# Plotting original series and differences
plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})

# Original Series
fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(dataset_train['Module_X_Lecture_attendance'])
ax1.set_title('Original Series')
ax1.axes.xaxis.set_visible(False)

# 1st Differencing
ax2.plot(dataset_train['Module_X_Lecture_attendance'].diff())
ax2.set_title('1st Order Differencing')
ax2.axes.xaxis.set_visible(False)

# 2nd Differencing
ax3.plot(dataset_train['Module_X_Lecture_attendance'].diff().diff())
ax3.set_title('2nd Order Differencing')

plt.show()

# Plotting ACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, (ax1, ax2, ax3) = plt.subplots(3)
plot_acf(dataset_train['Module_X_Lecture_attendance'], ax=ax1)
plot_acf(dataset_train['Module_X_Lecture_attendance'].diff().dropna(), ax=ax2)
plot_acf(dataset_train['Module_X_Lecture_attendance'].diff().diff().dropna(), ax=ax3)


fig, (ax1, ax2, ax3) = plt.subplots(3)
plot_pacf(dataset_train['Module_X_Lecture_attendance'], ax=ax1)
plot_pacf(dataset_train['Module_X_Lecture_attendance'].diff().dropna(), ax=ax2)
plot_pacf(dataset_train['Module_X_Lecture_attendance'].diff().diff().dropna(), ax=ax3)

plt.show()
