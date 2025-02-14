import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# 🔹 Load dataset
df = pd.read_csv("../data/processed_timetable_with_attendance_v7.csv")  # Replace with actual path
#df = df.drop(columns=['startTime', 'endTime'])

# # 🔹 Encode categorical variables
label_encoders = {}
categorical_columns = ['room', 'startTime', 'endTime']

# label_encoders = {}
# categorical_columns = ['room']



for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders for later use

# 🔹 Define features (X) and target variable (y)
X = df[['room', 'day', 'period', 'semester', 'partOfDay', 'startTime', 'endTime']]
#X = df[['room', 'day', 'period', 'semester', 'partOfDay']]
y = df['attendance']  # Binary: 0 (absent) or 1 (present)

# 🔹 Split into train & test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 🔹 Initialize & train Random Forest with class balancing
clf = RandomForestClassifier(n_estimators=200, max_depth=5, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# 🔹 Make predictions
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]  # Probability scores for ROC-AUC

# 🔹 Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# 🔹 Print results
print(f"✅ Accuracy: {accuracy:.2f}")
print(f"✅ Precision: {precision:.2f}")
print(f"✅ Recall: {recall:.2f}")
print(f"✅ F1-score: {f1:.2f}")
print(f"✅ ROC-AUC: {roc_auc:.2f}")
