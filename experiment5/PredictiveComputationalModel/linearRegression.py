#NOT UPDATED
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from single_lstms.variables import testing_set_size



def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Define dataset
combined_output = pd.read_csv('combined_output.csv', header=0)
print(combined_output)

X, y = combined_output.iloc[0:testing_set_size-12, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]].values, combined_output.iloc[0:testing_set_size-12, -1].values
print(X)

# Define the model
model = LinearRegression()

# Fit the model
model.fit(X, y)

# Test prediction with real data
test_X = combined_output.iloc[testing_set_size-12:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]].values
prediction = model.predict(test_X)


# Save predictions to CSV
np.savetxt('final_results.csv', prediction, delimiter=',', fmt='%f', header='FTSE100_Close_Price_Final_Prediction')

# Plot the prediction and real values
lstm1_predicted_stock_price = combined_output.iloc[testing_set_size-12:, 0:1].values
real_y = combined_output.iloc[testing_set_size-12:, -1].values
plt.rcParams["font.family"] = "Times New Roman"
plt.plot(real_y, color='black', label='Real Close Price')
plt.plot(prediction, color='green', label='MIPCM Predicted FTSE100 Close Price')
plt.plot(lstm1_predicted_stock_price, color='orange', label='Single LSTM Predicted FTSE100 Close Price')
plt.title('Real vs Predicted FTSE100 Close Price')
plt.xlabel('Months')
plt.ylabel('FTSE100 Close Price')
plt.legend()
plt.savefig('updated_final_analysis.png', dpi=1080, format='png')
plt.show()



predictions = [...]  # this should be your list of predictions
np_predictions = np.array(lstm1_predicted_stock_price)
# reshape it into a row vector
np_predictions = np_predictions.flatten()
print('FTSE100 Close Price Predicted via Single LSTM:', np_predictions)

# LSTM Result
# Mean Absolute Percentage Error
MAPE_lstm = calculate_mape(real_y, lstm1_predicted_stock_price)
print(f'LSTM MAPE: {MAPE_lstm:.3f}%')

print(f'FTSE100 Close Price Predicted via MIPCM: {prediction}')
# Final Result
# Mean Absolute Percentage Error
MAPE_final = calculate_mape(real_y, prediction)
print(f'MIPCM MAPE (no feature selection): {MAPE_final:.3f}%')