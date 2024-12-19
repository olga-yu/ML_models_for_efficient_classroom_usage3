#from https://github.com/Yerashenia/Predictive-Computational-Model-PCM/tree/master/Market_Index_Prediction

#NOT UPDATED
import pandas as pd

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from matplotlib import pyplot as plt
from single_lstms.variables import testing_set_size
from linear_regression_updated import MAPE_lstm




def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# define dataset
combined_output = pd.read_csv('combined_output.csv', header=0)
combined_output_neo4j = pd.read_csv('feature_selection_from_neo4j.csv', header=0)
print(combined_output_neo4j)

X, y = combined_output_neo4j.iloc[0:testing_set_size-12, :-1].values, combined_output_neo4j.iloc[0:testing_set_size-12, -1].values
print(X)

# define the model
model = LinearRegression()

# fit the model
model.fit(X, y)

# Test prediction with real data
test_X = combined_output_neo4j.iloc[testing_set_size-12:, :-1].values
prediction = model.predict(test_X)
print(f'FTSE100 Close Price predicted via MIPCM (with FS): {prediction}')

# Save predictions to csv
np.savetxt('final_results_feature_selection_neo4j.csv', prediction, delimiter=',', fmt='%f', header='FTSE100_Close_Price_Final_Prediction')

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
plt.savefig('updated_final_analysis_with_feature_regression_neo4j.png', dpi=1080, format='png')
plt.show()

# LSTM Result
# Mean Absolute Percentage Error
# MAPE_lstm = mean_absolute_percentage_error(real_y, lstm1_predicted_stock_price)
# print(f'LSTM MAPE: {MAPE_lstm:..3f}%')

# Print the MAPE value
print(f'LSTM MAPE: {MAPE_lstm:.3f}%')

# Final Result
# Mean Absolute Percentage Error
MAPE_final = calculate_mape(real_y, prediction)
print(f'MIPCM MAPE with Neo4j feature selection: {MAPE_final:.3f}%')