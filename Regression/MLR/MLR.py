'''
Multiple Linear MLR
- is there way to graphically represent other dependent attributes as well, like:
week, day, time_of_day, class_type, school, joint, degree, class_duration, enrollment, status?
- at the moment showing only 'faculty' VS (predicting) 'normalized attendance' on plot
'''

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

df = pd.read_csv(r'../../Data/output_lecture_seminar_processed3.csv', parse_dates=['date'])
X = df.drop(['Unnamed: 0', 'year', 'date','room_name', 'status', 'year','attendance', 'attendance_by_class', 'date-year', 'date-month', 'date-day','normalized_attendance','class_type_new',], axis=1)  #
print(X)

y = df['normalized_attendance']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = None)

mlr = LinearRegression() # #Fitting the Multiple Linear MLR model
mlr.fit(X_train, y_train)

# #Prediction of test set
y_pred_mlr= mlr.predict(X_test)

print("Intercept: ", mlr.intercept_)
print("Coefficients:")
list(zip(X, mlr.coef_))

#Predicted values
print("Prediction for test set: {}".format(y_pred_mlr))

#Actual value and the predicted value
mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
mlr_diff.head()


meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))

print('Root Mean Square Error:', rootMeanSqErr)
print('Mean Absolute Error:', meanAbErr)
print('R squared: {:.2f}'.format(mlr.score(X,y)*100))
print('Mean Square Error:', meanSqErr)

plt.scatter(df['faculty'], df['normalized_attendance'], color='red')
plt.title('faculty VS (predicting) normalized_attendance', fontsize=14)
plt.xlabel('faculty', fontsize=14)
plt.ylabel('normalized_attendance', fontsize=14)
plt.title('Multiple Linear MLR: 10 features')
plt.grid(True)
plt.show()








