from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

dataset = pd.read_csv('https://raw.githubusercontent.com/wiut-tutor2022/PHD_attendance_prediction/master/Data/output_lecture_seminar_processed3.csv', parse_dates=['date'])

feature_cols = ['week', 'day', 'time_of_day', 'class_type', 'faculty', 'school', 'joint', 'status', 'degree', 'enrollment', 'class_duration']

X = dataset[feature_cols]
y = dataset['normalized_attendance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = None)

mlr = LinearRegression() # Fitting the Multiple Linear MLR model
mlr.fit(X_train, y_train)

y_pred_mlr= mlr.predict(X_test)

print("Intercept: ", mlr.intercept_)
print("Coefficients:")
coef_df = pd.DataFrame({'feature': feature_cols, 'coefficient': abs(mlr.coef_)})
coef_df = coef_df.sort_values(by='coefficient', ascending=False)
print(coef_df)

meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))

print('Root Mean Square Error:', rootMeanSqErr)
print('Mean Absolute Error:', meanAbErr)
print('R squared: {:.2f}'.format(mlr.score(X,y)*100))
print('Mean Square Error:', meanSqErr)

plt.barh(coef_df['feature'], coef_df['coefficient'])
plt.title('Feature Importance for Multiple Linear MLR', fontsize=14)
plt.xlabel('Absolute Coefficient Value', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.grid(True)
plt.show()
