from pandas import read_csv, concat
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from matplotlib import pyplot as plt

#df = pd.read_csv(r'../data/output_14_18_7.csv')
df = pd.read_csv(r'../data/output_9_13_7.csv')

df.fillna(0, inplace=True)
df = df.drop('Extracted_Time', axis=1)
df = df.drop('User_sensor_mo.mean', axis=1)


###############333
'''
# Select the relevant features
feature_cols = ['week',  'school',  'enrollment', ]

# Extract the features and target variable
X = df[feature_cols]
y = df['normalized_attendance2'] # this is classified normalized attendance, normalized attendance is room_attendance/course_enrollment
##################3333

training_data = df.drop('Class',axis=1)
training_output = df['Class']
training_data_cols = np.array(df.drop('Class',axis=1).columns.values.tolist())
'''
training_data = df.drop('Time', axis=1)

training_output = df['sensor_mo.mean']
training_data_cols = np.array(df.drop('sensor_mo.mean',axis=1).columns.values.tolist())

# Check the shape of your data and target variable
print(training_data.shape)  # Should match the number of samples and features
print(training_output.shape)  # Should match the number of samples

# Check the types of your columns to ensure they are numerical
print(training_data.dtypes)

# Check for missing values or non-numeric columns
print(training_data.isnull().sum())  # Ensure there are no missing values in your features
print(training_output.isnull().sum())  # Ensure there are no missing values in your target



# Fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(training_data, training_output)

# Display the relative importance of each attribute
sorted_idx = model.feature_importances_.argsort()

print(model.feature_importances_)
weights = np.array(model.feature_importances_)
training_data_cols_matrix = np.expand_dims(training_data_cols, axis=1)
weights = np.expand_dims(weights, axis=1)

# Table output
table = np.concatenate([training_data_cols_matrix, weights], axis=1)
table = pd.DataFrame(table)
table.columns = ['Attribute', 'Weights']
table.to_csv('weights_importance_uc1.csv', index=False)
print(table)

# Plot output
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'figure.figsize': (12.0, 8.0)})
plt.rcParams.update({'font.size': 14})
sorted_idx = model.feature_importances_.argsort()
plt.barh(training_data_cols[sorted_idx], model.feature_importances_[sorted_idx], color='green')
plt.xlabel("Random Forest Feature Importance")
plt.savefig('RFFI_UC1.png', dpi=1080, format='png')
plt.show()