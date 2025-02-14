import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import skfuzzy as fuzz

# 🔹 Load dataset
dataset = pd.read_csv("data/processed_motionData2025_3.csv")  # Replace with actual file

# 🔹 Define features (X) and target variable (y)
X = dataset[["StudentID", "TimePeriod", "date-year", "date-month", "date-day", "Season", "Weekday", "Semester"]]
y = dataset["sensor_mo.mean"]  # Binary classification (0 or 1)

# 🔹 Apply Fuzzy Logic to `sensor_mo.mean` for better class distribution
# Membership functions for Low, Medium, High Activity
y_low = fuzz.membership.trimf(y, [0, 0, 0.5])  # Low Activity
y_med = fuzz.membership.trimf(y, [0, 0.5, 1])  # Medium Activity
y_high = fuzz.membership.trimf(y, [0.5, 1, 1])  # High Activity

# Assign fuzzy values instead of strict 0/1
y_fuzzy = (y_low * 0 + y_med * 0.5 + y_high * 1)

# 🔹 Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_fuzzy, test_size=0.2, random_state=42, stratify=y)

# 🔹 Standardize numeric data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🔹 Build Neural Network model
model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),  # Input layer
    keras.layers.Dense(64, activation='relu'),  # First hidden layer
    keras.layers.Dense(32, activation='relu'),  # Second hidden layer
    keras.layers.Dense(16, activation='relu'),  # Third hidden layer
    keras.layers.Dense(1, activation='sigmoid')  # Output layer (fuzzy classification)
])

# 🔹 Custom Loss Function for Fuzzy Learning
def fuzzy_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))  # Mean Squared Error (MSE) for fuzzy learning

# 🔹 Compile model
model.compile(optimizer='adam', loss=fuzzy_loss, metrics=['mae'])  # Using Mean Absolute Error (MAE) for fuzziness

# 🔹 Train model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# 🔹 Make predictions
y_pred_fuzzy = model.predict(X_test)

# 🔹 Convert Fuzzy Predictions to Hard Labels
y_pred = (y_pred_fuzzy > 0.5).astype(int)

# 🔹 Evaluate performance
accuracy = accuracy_score(y_test.round(), y_pred)
precision = precision_score(y_test.round(), y_pred)
recall = recall_score(y_test.round(), y_pred)
f1 = f1_score(y_test.round(), y_pred)
roc_auc = roc_auc_score(y_test, y_pred_fuzzy)

# 🔹 Print results
print(f"✅ Accuracy: {accuracy:.2f}")
print(f"✅ Precision: {precision:.2f}")
print(f"✅ Recall: {recall:.2f} 🔥")
print(f"✅ F1-score: {f1:.2f}")
print(f"✅ ROC-AUC: {roc_auc:.2f} 🚀")
