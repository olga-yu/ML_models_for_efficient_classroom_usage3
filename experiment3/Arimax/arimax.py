import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler

def calculate_mape(y_true, y_pred):
    percentage_errors = np.abs((y_true - y_pred) / y_true)
    mape = np.mean(percentage_errors) * 100
    return mape

testing_set_size = 12

# Load and preprocess the training dataset
dataset_train = pd.read_csv('../../Data/westminster.csv', header=0, index_col=0)
training_set = dataset_train.iloc[6:, 0:1].values


#dataset_train = pd.read_csv('Processed_Input_Data_FTSE100_1985_21.csv', header=0, index_col=0)
#training_set = dataset_train.iloc[60:, 0:1].values

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []
for i in range(0, (len(training_set) - testing_set_size)):
    X_train.append(training_set_scaled[i:i+1, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# ARIMAX model configuration
order = (2, 0, 1)  # ARIMA order
exog = X_train  # exogenous variables

model = ARIMA(endog=y_train, exog=exog, order=order)
arimax_pred = model.fit()

# Load and preprocess the testing dataset (last 12 months)
dataset_test = pd.read_csv('../../Data/westminster.csv', header=0, index_col=0)

real_stock_price = dataset_test.iloc[-testing_set_size:, 0:1].values

dataset_total = pd.concat((dataset_train['normalized_attendance'], dataset_test['normalized_attendance']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test):].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range((len(training_set) - testing_set_size), len(training_set)):
    X_test.append(inputs[i:i+1, 0])
X_test = np.array(X_test)

# Make predictions with ARIMAX model for the last 12 months
predicted_stock_price = arimax_pred.predict(start=len(y_train), end=len(y_train) + testing_set_size - 1, exog=X_test)

# Rescale the predicted prices
predicted_stock_price = sc.inverse_transform(predicted_stock_price.reshape(-1, 1))

# Calculate MAPE
mape = calculate_mape(real_stock_price, predicted_stock_price)
print(f'Single FTSE100 ARIMAX MAPE: {mape:.3f}%')