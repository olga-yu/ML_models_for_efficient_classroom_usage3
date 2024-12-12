import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import matplotlib.pyplot as plt

# Generate synthetic dataset
X, y = make_classification(
    n_features=6,
    n_classes=3,
    n_samples=800,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1,
)

# Visualize the dataset
plt.scatter(X[:, 0], X[:, 1], c=y, marker="*")
plt.title('Scatter plot of generated dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=125
)

# Build a Gaussian Naive Bayes Classifier
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Predict on the test dataset
y_pred = model.predict(X_test)

# Evaluate classification metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

# For regression-like comparison
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the metrics
print("Classification Report Metrics:")
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("\nRegression Report Metrics:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared: {r2}")
