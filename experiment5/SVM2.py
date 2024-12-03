import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pandas as pd

from sklearn.decomposition import PCA

# Load dataset
dataset = pd.read_csv('data/output_9_18.csv', parse_dates=['Time'])

# Convert 'Time' to datetime
dataset['Time'] = pd.to_datetime(dataset['Time'], format='%m/%d/%Y %H:%M')

# Extract hour and minute
dataset['Hour'] = dataset['Time'].dt.hour
dataset['Minute'] = dataset['Time'].dt.minute

# Select only two features for simplicity
X = dataset[['Hour', 'WeekDay']]  # For visualization, we use 'Hour' and 'WeekDay' (2D)
y = dataset['sensor_mo.mean']  # Target variable (attendance)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train_scaled, y_train)

# Create a mesh grid for plotting the decision boundary
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict the class labels for each point in the mesh grid
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.Spectral)

# Scatter plot of the training data points
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, s=20, edgecolors="k", cmap=plt.cm.Spectral)
plt.xlabel('Hour')
plt.ylabel('WeekDay')
plt.title('SVM Decision Boundary (RBF kernel)')

# Show the plot
plt.show()
