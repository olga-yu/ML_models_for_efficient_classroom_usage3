# the experiment with Cost of missing attendance

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller

# Absolute Percentage Error (or simply MAPE)
def calculate_mape(y_true, y_pred):
    percentage_errors = np.abs((y_true - y_pred) / y_true)
    mape = np.mean(percentage_errors) * 100
    return mape

# Load and preprocess the training dataset
dataset_train = pd.read_csv('../../Data/2017_sem2_attendance_data.csv', header=0, index_col=0)
training_set = dataset_train.iloc[700:, 17:].values #takes 1/3 of attendance from data and column of data
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
dataset_test = pd.read_csv('../../Data/2017_sem2_attendance_data.csv', header=0, index_col=0)
attendance = dataset_test.iloc[-testing_set_size:, 0:1].values

dataset_total = pd.concat((dataset_train['attendance'], dataset_test['attendance']), axis=0)
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
result = adfuller(dataset_train['attendance'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# Plotting original series and differences
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})

# Original Series
fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(dataset_train['attendance'])
ax1.set_title('Original Series')
ax1.axes.xaxis.set_visible(False)

# 1st Differencing
ax2.plot(dataset_train['attendance'].diff())
ax2.set_title('1st Order Differencing')
ax2.axes.xaxis.set_visible(False)

# 2nd Differencing
ax3.plot(dataset_train['attendance'].diff().diff())
ax3.set_title('2nd Order Differencing')

plt.show()

# Plotting ACF
from statsmodels.graphics.tsaplots import plot_acf

fig, (ax1, ax2, ax3) = plt.subplots(3)
plot_acf(dataset_train['attendance'], ax=ax1)
plot_acf(dataset_train['attendance'].diff().dropna(), ax=ax2)
plot_acf(dataset_train['attendance'].diff().diff().dropna(), ax=ax3)

plt.show()
