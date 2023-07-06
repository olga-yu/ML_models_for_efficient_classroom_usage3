import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Load the dataset
dataset = pd.read_csv('output_lecture_seminar_processed3.csv', parse_dates=['date'])

# Select the relevant features
feature_cols = ['week', 'day', 'time_of_day', 'class_type', 'faculty', 'school', 'joint', 'status', 'degree', 'enrollment', 'class_duration', 'joint', 'status','time_of_day','day']

# Extract the features and target variable
X = dataset[feature_cols]
y = dataset['normalized_attendance']

# Fit the random forest regression model
regr = RandomForestRegressor(n_estimators=100, max_depth=5)
regr.fit(X, y)

# Find the importance of each feature/column
importance = regr.feature_importances_

# Print the feature importance scores
for feature, score in zip(feature_cols, importance):
    print(feature, score)