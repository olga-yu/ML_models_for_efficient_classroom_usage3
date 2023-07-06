import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler

def calculate_mape(y_true, y_pred):
    percentage_errors = np.abs((y_true - y_pred) / y_true)
    mape = np.mean(percentage_errors) * 100
    return mape

testing_set_size = 3

# Load and preprocess the training dataset
dataset_train = pd.read_csv('../../Data/westminster.csv', header=0, index_col=0)
training_set = dataset_train.iloc[6:, 0:1].values

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

X_train = training_set_scaled[:-testing_set_size]
y_train = training_set_scaled[testing_set_size:]

# ARIMAX model configuration
order = (2, 0, 1)  # ARIMA order
exog = X_train  # exogenous variables

model = ARIMA(endog=y_train, exog=exog, order=order)
arimax_pred = model.fit()

# Load and preprocess the testing dataset (last 12 months)
dataset_test = pd.read_csv('../../Data/westminster.csv', header=0, index_col=0)
real_attendance_price = dataset_test.iloc[-testing_set_size:, 0:1].values
#predict the price of paying for contract = 33,000,000

dataset_total = pd.concat((dataset_train['Module_X_Lecture_attendance'], dataset_test['Module_X_Lecture_attendance']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test):].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = inputs[-testing_set_size:]

# Make predictions with ARIMAX model for the last 12 months
predicted_attendance_price = arimax_pred.predict(start=len(y_train), end=len(y_train) + testing_set_size - 1, exog=X_test)

# Rescale the predicted prices
predicted_attendance_price = sc.inverse_transform(predicted_attendance_price.reshape(-1, 1))

# Calculate MAPE
mape = calculate_mape(real_attendance_price, predicted_attendance_price)
print(f'Attendance ARIMAX MAPE: {mape:.3f}%')
