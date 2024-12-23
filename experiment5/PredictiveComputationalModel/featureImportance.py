#NOT UPDATED
# Feature Importance
from pandas import read_csv
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from single_lstms.variables import testing_set_size
from matplotlib import pyplot as plt

# load the iris datasets
training_data_cols = read_csv('combined_output.csv').columns.values
training_data_cols = np.array(training_data_cols[0:len(training_data_cols) - 1].tolist())

training_data = pd.read_csv('combined_output.csv', header=0).values
training_data = training_data[:, :-1]

training_output = read_csv('Processed_Input_Data_FTSE100_1985_21.csv', header=0, index_col=0).values
training_output = training_output[len(training_output) - testing_set_size:, 0]
# fit an Extra Trees model to the data
model = ExtraTreesRegressor(n_estimators=100)
model.fit(training_data, training_output)
# display the relative importance of each attribute
weights = np.array(model.feature_importances_)
training_data_cols_matrix = np.expand_dims(training_data_cols, axis=1)
weights = np.expand_dims(weights, axis=1)

# Table output
table = np.concatenate([training_data_cols_matrix, weights], axis=1)
table = pd.DataFrame(table)
table.columns = ['Attribute', 'Weights']
table.to_csv('weights_importance_uc2.csv', index=False)
print(table)

# Plot output
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'figure.figsize': (12.0, 8.0)})
plt.rcParams.update({'font.size': 14})
sorted_idx = model.feature_importances_.argsort()
plt.barh(training_data_cols[sorted_idx], model.feature_importances_[sorted_idx], color='green')
plt.xlabel("Random Forest Feature Importance")
plt.savefig('RFFI_UC2.png', dpi=1080, format='png')
plt.show()