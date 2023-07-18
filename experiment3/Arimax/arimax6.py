import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm

from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler


from statsmodels.tsa.stattools import adfuller


# we will see how to compute one of the methods to determine forecast accuracy
# Absolute Percentage Error (or simply MAPE)
def calculate_mape(y_true, y_pred):
    percentage_errors = np.abs((y_true - y_pred) / y_true)
    mape = np.mean(percentage_errors) * 100
    return mape

#
testing_set_size = 3

# Load and preprocess the training dataset

#dataset_train = pd.read_csv('../../Data/westminster.csv', header=0, index_col=0) # returns first row returns row from which to start and index_col = 0 returns first column
dataset_train = pd.read_csv('https://raw.githubusercontent.com/wiut-tutor2022/ML_models_for_efficient_classroom_usage3/blob/master/Data/westminster.csv', header=0, index_col=0) # returns first row returns row from which to start and index_col = 0 returns first column
training_set = dataset_train.iloc[3:, 0:1].values
#training_set2 = dataset_train.iloc[2:3, 0:3].values # rows to select and column to select
print("training set")
print(training_set)


# print (f"Total samples: {len (df)}")
print("dataset_train")
print (dataset_train)
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
print("training set scaled")
print(training_set_scaled)

X_train = []
y_train = []

for i in range(0, (len(training_set) - testing_set_size)):
    X_train.append(training_set_scaled[i:i+1, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# ARIMAX model configuration
order = (1, 1, 1)  # ARIMA order # Autoregressive, Differencing, Moving Average orders , did not understand fully p, d, q values, https://analyticsindiamag.com/quick-way-to-find-p-d-and-q-values-for-arima/


exog = X_train  # exogenous variables

model = ARIMA(endog=y_train, exog=exog, order=order)
arimax_pred = model.fit() # Model fitting is a measure of how well a machine learning model generalizes to similar data to
# that on which it was trained. A model that is well-fitted produces more accurate outcomes.

# Load and preprocess the testing dataset (last 12 months)
dataset_test = pd.read_csv('https://raw.githubusercontent.com/wiut-tutor2022/ML_models_for_efficient_classroom_usage3/blob/master/Data/westminster.csv', header=0, index_col=0)

attendance = dataset_test.iloc[-testing_set_size:, 0:1].values
print('real_attendance')
print(attendance)

print('print dataset Test')
print(len(dataset_test))

dataset_total = pd.concat((dataset_train['Module_X_Lecture_attendance'], dataset_test['Module_X_Lecture_attendance']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test):].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range((len(training_set) - testing_set_size), len(training_set)):
    X_test.append(inputs[i:i+1, 0])
X_test = np.array(X_test)


# Make predictions with ARIMAX model for the last 12 months
predicted_attendance = arimax_pred.predict(start=len(y_train), end=len(y_train) + testing_set_size - 1, exog=X_test)

# Rescale the predicted prices
predicted_attendance = sc.inverse_transform(predicted_attendance.reshape(-1, 1))

# Calculate MAPE
mape = calculate_mape(attendance, predicted_attendance)
print(f'Single Attendance ARIMAX MAPE: {mape:.3f}%')

#to check to find p, d, q values of Arima, based on https://analyticsindiamag.com/quick-way-to-find-p-d-and-q-values-for-arima/
#finding the P value
from statsmodels.tsa.stattools import adfuller
result = adfuller(dataset_train['Module_X_Lecture_attendance'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
  print('\t%s: %.3f' % (key, value))

print("p-value greater ")



plt.plot(dataset_train['Module_X_Lecture_attendance'], color='green',  label='Module Lecture attendance')
plt.show()  #difficult to be judged as stationary or non-stationary as not much data is given. So far data provided
# gives assumptions for it to be non-stationary thus using

#

#finding d

import numpy as np, pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})

# Original Series
fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(dataset_train.Module_X_Lecture_attendance);
ax1.set_title('Original Series');
ax1.axes.xaxis.set_visible(False)
# 1st Differencing
ax2.plot(dataset_train.Module_X_Lecture_attendance.diff());
ax2.set_title('1st Order Differencing');
ax2.axes.xaxis.set_visible(False)
# 2nd Differencing
ax3.plot(dataset_train.Module_X_Lecture_attendance.diff().diff());
ax3.set_title('2nd Order Differencing')
plt.show()

acf_original = plot_acf(dataset_train)


from statsmodels.graphics.tsaplots import plot_acf
fig, (ax1, ax2, ax3) = plt.subplots(3)
plot_acf(dataset_train.Module_X_Lecture_attendance, ax=ax1)
plot_acf(dataset_train.Module_X_Lecture_attendance.diff().dropna(), ax=ax2)
plot_acf(dataset_train.Module_X_Lecture_attendance.diff().diff().dropna(), ax=ax3)
