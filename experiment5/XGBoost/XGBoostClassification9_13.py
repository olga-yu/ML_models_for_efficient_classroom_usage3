import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv('../data/output_9_13_7.csv')

# Process 'Extracted_Time' to create new features (e.g., PartOfDay)
def process_time(time_str):
    hour = int(time_str.split(':')[0])
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

df['PartOfDay'] = df['Extracted_Time'].apply(process_time)
df['PartOfDay'] = df['PartOfDay'].astype('category').cat.codes

# Drop unnecessary columns
df = df.drop(columns=['Time', 'Extracted_Time'])

# Split into features (X) and target (y)
X = df.drop(columns=['sensor_mo.mean'])
y = df['sensor_mo.mean']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize XGBoost classifier
xgb_model = XGBClassifier(scale_pos_weight=4, random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions (get probabilities as well)
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]  # Probability of class 1

# Calculate Classification Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Convert predictions to continuous form to use regression metrics
rmse = mean_squared_error(y_test, y_pred_proba, squared=False)
mae = mean_absolute_error(y_test, y_pred_proba)
mse = mean_squared_error(y_test, y_pred_proba)
r2 = r2_score(y_test, y_pred_proba)

# Print Regression-like metrics
print("\nRegression-like Metrics:")
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared:", r2)
