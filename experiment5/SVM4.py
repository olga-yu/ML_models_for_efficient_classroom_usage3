import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler  # Updated scaler
from sklearn.model_selection import train_test_split

# Load dataset
dataset = pd.read_csv('data/output_9_18.csv', parse_dates=['Time'])



# Convert 'Time' to datetime
dataset['Time'] = pd.to_datetime(dataset['Time'], format='%m/%d/%Y %H:%M')

# Extract hour and minute
dataset['Hour'] = dataset['Time'].dt.hour
dataset['Minute'] = dataset['Time'].dt.minute




# Convert sensor_mo.mean to categorical (example with binning)
bins = [0, 10, 20, 30]
labels = [0, 1, 2]
dataset['sensor_mo.mean'] = pd.cut(dataset['sensor_mo.mean'], bins=bins, labels=labels)

# Ensure features have variability
print(dataset[['Hour', 'WeekDay']].describe())

# Visualize feature distribution
import seaborn as sns
sns.scatterplot(x='Hour', y='WeekDay', hue='sensor_mo.mean', data=dataset)
plt.show()

# Train SVM with scaled data
X = dataset[['Hour', 'WeekDay']]  # Replace 'WeekDay' if necessary
y = dataset['sensor_mo.mean']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC(kernel='rbf', C=10.0, gamma=0.1)
svm.fit(X_train_scaled, y_train)

# Plot decision boundary
x_min, x_max = X_train_scaled[:, 0].min() - 0.1, X_train_scaled[:, 0].max() + 0.1
y_min, y_max = X_train_scaled[:, 1].min() - 0.1, X_train_scaled[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.Spectral)
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, s=20, edgecolors="k", cmap=plt.cm.Spectral)
plt.xlabel('Hour')
plt.ylabel('WeekDay')
plt.title('Improved SVM Decision Boundary')
plt.show()
