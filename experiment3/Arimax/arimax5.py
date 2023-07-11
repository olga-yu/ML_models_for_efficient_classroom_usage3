import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler


from statsmodels.tsa.stattools import adfuller


# we will see how to compute one of the methods to determine forecast accuracy
# Absolute Percentage Error (or simply MAPE)
# def calculate_mape(y_true, y_pred):
#     percentage_errors = np.abs((y_true - y_pred) / y_true)
#     mape = np.mean(percentage_errors) * 100
#     return mape
#
#
# testing_set_size = 3

# Load and preprocess the training dataset
df = pd.read_csv('../../Data/westminster.csv', names = ['Date'], parse_dates=['Date'], index='Date')
# print (f"Total samples: {len (df)}")
# print (df.head)


fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,8))
plot_acf(df, ax=ax1, lags=20)
plot_pacf(df, ax2, lags=20)

plt.show()
# plot_acf(df.value)
# f = plt.figure()
# ax1 = f.add_subplot(121)
# ax1.set_title('1st order differencing')
# ax1.plot(df.value.diff())
#
# ax2 = f.add_subplot(122)
# plot_acf(df.value.diff().dropna(), ax=ax2)
# plt.show()

# dataset_train = pd.read_csv('../../Data/westminster.csv', header=0, index_col=0)
#
# training_set = dataset_train.iloc[3:, 0:1].values
# print('training set')
# print(len(training_set))
#
# # shows 75% of values and works on "Module_X_Lecture_attendance"
#
#
# sc = MinMaxScaler(feature_range=(0, 1))
# training_set_scaled = sc.fit_transform(training_set)
#
# print("training set scaled")
# print(training_set_scaled)
#
# X_train = []
# y_train = []
# for i in range(0, (len(training_set) - testing_set_size)):
#     X_train.append(training_set_scaled[i:i + 1, 0])
#     y_train.append(training_set_scaled[i, 0])
# X_train, y_train = np.array(X_train), np.array(y_train)
#
# # ARIMAX model configuration
# order = (2, 0, 1)  # ARIMA order
# exog = X_train  # exogenous variables
#
# model = ARIMA(endog=y_train, exog=exog, order=order)
# arimax_pred = model.fit()
#
# # Load and preprocess the testing dataset (last 12 months)
# dataset_test = pd.read_csv('../../Data/westminster.csv', header=0, index_col=0)
# real_attendance = dataset_test.iloc[-testing_set_size:, 0:1].values
# print("real 10% of attendance")
# print(real_attendance)
# # predict the price of paying for contract = 33,000,000
#
# dataset_total = pd.concat((dataset_train['Module_X_Lecture_attendance'], dataset_test['Module_X_Lecture_attendance']),
#                           axis=0)
# inputs = dataset_total[len(dataset_total) - len(dataset_test):].values
# inputs = inputs.reshape(-1, 1)
# inputs = sc.transform(inputs)
# X_test = inputs[-testing_set_size:]
#
# # Make predictions with ARIMAX model for the last 12 months
# predicted_attendance_price = arimax_pred.predict(start=len(y_train), end=len(y_train) + testing_set_size - 1,
#                                                  exog=X_test)
#
# # Rescale the predicted prices
# predicted_attendance_price = sc.inverse_transform(predicted_attendance_price.reshape(-1, 1))
#
# # Calculate MAPE
# mape = calculate_mape(real_attendance, predicted_attendance_price)
# print(f'Attendance ARIMAX MAPE: {mape:.3f}%')
