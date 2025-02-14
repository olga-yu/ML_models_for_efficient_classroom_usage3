import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

# Load dataset
dataset = pd.read_csv('data/processed_motionData2025_3.csv')

# Select features and target variable
X = dataset[["StudentID", "TimePeriod", 'date-year', 'date-month', 'date-day', 'Season', 'Weekday', 'Semester']]
y = dataset['sensor_mo.mean']  # Target variable (0 or 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate scale_pos_weight (ratio of 0s to 1s)
scale_pos_weight = (y_train.value_counts()[0] / y_train.value_counts()[1])

# Initialize XGBoost Classifier with imbalance handling
clf = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    scale_pos_weight=scale_pos_weight,  # Balances class weights
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

# Train the model
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]  # Probability scores for ROC-AUC

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# Print results
print(f"✅ Accuracy: {accuracy:.2f}")
print(f"✅ Precision: {precision:.2f}")
print(f"✅ Recall: {recall:.2f}")
print(f"✅ F1-score: {f1:.2f}")
print(f"✅ ROC-AUC: {roc_auc:.2f}")
