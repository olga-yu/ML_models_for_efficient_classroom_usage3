import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score

df = pd.read_csv(r'..\Data\output_lecture_seminar_processed4.csv', parse_dates=['date'])
df.fillna(0, inplace=True)
print(df.shape)

#split data into inputs and targets
#X = df.drop(['Unnamed: 0', 'year', 'date','room_name', 'status', 'year','attendance', 'attendance_by_class', 'date-year', 'date-month', 'date-day','normalized_attendance','normalized_attendance2','class_type_new',], axis=1)  #
# Select the relevant features
feature_cols = ['week',  'school',  'enrollment', 'day', 'faculty']

# Extract the features and target variable
X = df[feature_cols]

y = df['normalized_attendance2']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)

#another method of creating a NN (not from scatch) using MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(20,), activation='logistic', solver='adam', max_iter=1500)
mlp.fit(X_train,y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

print(confusion_matrix(y_train,predict_train))
print(classification_report(y_train,predict_train))