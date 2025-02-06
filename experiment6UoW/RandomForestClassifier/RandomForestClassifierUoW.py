import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_curve
import seaborn as sns  # For better visualization of the confusion matrix

# Load dataset
dataset = pd.read_csv('../data/UoWprocessed_motionData2025_3.csv')

# Drop unnecessary columns safely
dataset = dataset.drop(columns=['Time', 'Date'], errors='ignore')

# Select features and target variable
X = dataset[["StudentID", "TimePeriod", 'date-year', 'date-month', 'date-day', 'Season', 'Weekday', 'Semester']]
y = dataset['sensor_mo.mean']  # Ensure this column exists

# Step 2: Split the data **with stratification**
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Check class distribution before SMOTE
print("Before SMOTE:")
print(y_train.value_counts())

# Ensure y_train is an integer and binary for SMOTE
y_train = y_train.astype(int)

# Step 3: Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check class distribution after SMOTE
print("After SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

# Train Random Forest Classifier (not Regressor)
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train_resampled, y_train_resampled)

# Predictions (using classification probabilities)
y_probs = rf.predict_proba(X_test)[:, 1]  # Probability of class 1

# Step 4: Adjust decision threshold using Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
best_threshold = thresholds[np.argmax(precision + recall)]  # Best precision-recall tradeoff
print(f"Best Threshold: {best_threshold}")

# Apply the new threshold
y_pred_adjusted = (y_probs >= best_threshold).astype(int)

# Model Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred_adjusted))
# print("Classification Report:")


# Step 6: Classification Report with zero_division
print("Classification Report:")
print(classification_report(y_test, y_pred_adjusted, zero_division=1))  # Use zero_division=1 to return 1 when precision is undefined


# Step 5: Confusion Matrix
cm = confusion_matrix(y_test, y_pred_adjusted)
print("Confusion Matrix:")
print(cm)

# Step 6: Visualize Confusion Matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
