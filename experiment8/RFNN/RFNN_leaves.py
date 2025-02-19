from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

# Load your dataset
dataset = pd.read_csv('../data/processed_motionData2025_3.csv')

# Select features (X) and target variable (y)
X = dataset[["StudentID", "TimePeriod", 'date-year', 'date-month', 'date-day', 'Season', 'Weekday', 'Semester']]
y = dataset['sensor_mo.mean']  # Ensure this column contains 0s and 1s

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest with the best class weight found earlier
rf = RandomForestClassifier(n_estimators=100, class_weight={0: 1, 1: 3}, random_state=42)
rf.fit(X_train, y_train)

# Extract leaf indices
X_train_leaves = rf.apply(X_train)  # Get leaf indices for training
X_test_leaves = rf.apply(X_test)  # Get leaf indices for testing

# One-hot encode the leaf indices
encoder = OneHotEncoder()
X_train_encoded = encoder.fit_transform(X_train_leaves).toarray()
X_test_encoded = encoder.transform(X_test_leaves).toarray()

# Combine original features with Random Forest leaf features
X_train_combined = np.hstack([X_train, X_train_encoded])
X_test_combined = np.hstack([X_test, X_test_encoded])

print("✅ Random Forest leaf features added. Shape:", X_train_combined.shape)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Define Neural Network
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_combined.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification output
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

# Train model
model.fit(X_train_combined, y_train, validation_data=(X_test_combined, y_test), epochs=20, batch_size=32)
