import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE  # Import SMOTE for balancing

# 🔹 Load dataset
df = pd.read_csv("../data/processed_timetable_with_attendance_v7.csv")  # Replace with actual file

# 🔹 Encode categorical variables
label_encoders = {}
categorical_columns = ['room', 'startTime', 'endTime']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders for later use

# 🔹 Define features (X) and target variable (y)
X = df[['room', 'day', 'period', 'semester', 'partOfDay', 'startTime', 'endTime']]
y = df['attendance']  # Binary: 0 (absent) or 1 (present)

# 🔹 Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 🔹 Apply SMOTE to balance dataset
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 🔹 Initialize & train Random Forest with optimized hyperparameters
clf = RandomForestClassifier(
    n_estimators=300,        # More trees for better performance
    max_depth=12,            # Allow deeper trees for better learning
    min_samples_split=5,     # Prevent overfitting
    class_weight='balanced', # Handle imbalanced classes
    random_state=42
)
clf.fit(X_train_resampled, y_train_resampled)

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
print(f"✅ Recall: {recall:.2f}  ")
print(f"✅ F1-score: {f1:.2f}")
print(f"✅ ROC-AUC: {roc_auc:.2f} ")
