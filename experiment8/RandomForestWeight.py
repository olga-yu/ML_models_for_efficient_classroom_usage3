from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

# Load your dataset
dataset = pd.read_csv('data/processed_motionData2025_3.csv')

# Select features (X) and target variable (y)
X = dataset[["StudentID", "TimePeriod", 'date-year', 'date-month', 'date-day', 'Season', 'Weekday', 'Semester']]
y = dataset['sensor_mo.mean']  # Ensure this column contains 0s and 1s

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize variables to track best weight
best_auc = 0
best_weights = None

# Try different weight settings for class 1
for w in [1, 2, 3, 4, 5]:  # You can adjust this range
    class_weights = {0: 1, 1: w}  # Giving higher weight to class 1

    # Train Random Forest with class weights
    clf = RandomForestClassifier(n_estimators=100, class_weight=class_weights, random_state=42)
    clf.fit(X_train, y_train)

    # Predict probabilities for ROC-AUC calculation
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)  # Get binary predictions

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Print results for each weight setting
    print(f"Class Weight {class_weights} → ROC-AUC: {roc_auc:.3f}, Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")

    # Keep track of the best weight setting
    if roc_auc > best_auc:
        best_auc = roc_auc
        best_weights = class_weights

print(f"\n🔹 Best class weights: {best_weights} with ROC-AUC: {best_auc:.3f}")

# Train final model with the best class weight
final_clf = RandomForestClassifier(n_estimators=100, class_weight=best_weights, random_state=42)
final_clf.fit(X_train, y_train)

# Final predictions
y_pred_final = final_clf.predict(X_test)
y_prob_final = final_clf.predict_proba(X_test)[:, 1]

# Final performance metrics
final_accuracy = accuracy_score(y_test, y_pred_final)
final_precision = precision_score(y_test, y_pred_final)
final_recall = recall_score(y_test, y_pred_final)
final_f1 = f1_score(y_test, y_pred_final)
final_roc_auc = roc_auc_score(y_test, y_prob_final)

print("\n📌 **Final Model Performance with Best Class Weights:**")
print(f"✅ Accuracy: {final_accuracy:.2f}")
print(f"✅ Precision: {final_precision:.2f}")
print(f"✅ Recall: {final_recall:.2f}")
print(f"✅ F1-score: {final_f1:.2f}")
print(f"✅ ROC-AUC: {final_roc_auc:.3f}")
