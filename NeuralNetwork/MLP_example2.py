import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score

#df = pd.read_csv('/Users/nataliayerashenia/PycharmProjects/DMML_T9_NN/T4_diabetes.csv')
df = pd.read_csv(r'..\Data\output_lecture_seminar_processed4.csv', parse_dates=['date'])
print(df.shape)

#split data into inputs and targets

# Select the relevant features
feature_cols = ['week',  'school',  'enrollment', 'day', 'faculty']

# Extract the features and target variable
X = df[feature_cols]
y = df['normalized_attendance2'] # this is classified normalized attendance, normalized attendance is room_attendance/course_enrollment

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
mlp.fit(X_train,y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

print(confusion_matrix(y_train,predict_train))
print(classification_report(y_train,predict_train))

#added #########################

y_pred = mlp.predict(X_test)

from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(y_test, y_pred))

df = pd.DataFrame({'Real Values':y_test, 'Predicted Values':y_pred})
print(df)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))



plt.plot(mlp.loss_curve_)
plt.title("Training Loss Curve for MLP Classifier: 5 parameters")
plt.xlabel("Iteration")
plt.ylabel("Normalized attendance")
plt.show()