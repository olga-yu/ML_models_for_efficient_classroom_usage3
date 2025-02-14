from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load dataset
dataset = pd.read_csv('data/processed_motionData2025_3.csv')

# Select features and target variable
X = dataset[["StudentID", "TimePeriod", 'date-year', 'date-month', 'date-day', 'Season', 'Weekday', 'Semester']]
y = dataset['sensor_mo.mean']  # Ensure this column exists and contains only 0 or 1

# Convert categorical features to numeric (if needed)
X = pd.get_dummies(X, columns=['Season', 'Weekday', 'Semester'], drop_first=True)

# Drop StudentID if it's not relevant
X = X.drop(columns=['StudentID'], errors='ignore')

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Support Vector Classifier

clf = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)

clf.fit(X_train, y_train)

# Predict on test data
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]  # FIX: Now works correctly

# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# Print Metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
print(f"ROC-AUC: {roc_auc:.2f}")
