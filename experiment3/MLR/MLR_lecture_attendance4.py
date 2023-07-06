from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error, mean_absolute_error
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics

# Load the dataset from a CSV file#

dataset = pd.read_csv(r'../../Data/westminster.csv')
feature_cols = ['Week', 'Reg', 'Campus_Lecture_attendance', 'Module_A_deadlines',
                'Module_B_deadlines', 'Module_C_deadlines', 'L_timeslot 9-11', 'L_timeslot 11-13', 'L_timeslot 14-16']
# need to select only relevant labels
# Select the relevant features

# Extract the features and target variable
X = dataset[feature_cols]
print(X)
y = dataset['L_timeslot 11-13']


train_data = pd.read_csv('../../Data/westminster.csv')
test_data = pd.read_csv('../../Data/westminster3.csv')

X_train = train_data.drop('L_timeslot 11-13', axis=1)
y_train = train_data['L_timeslot 11-13']

X_test = test_data.drop('L_timeslot 11-13', axis=1)
y_test = test_data['L_timeslot 11-13']

# Fit the Multiple Linear MLR model
mlr = LinearRegression()
mlr.fit(X_train, y_train)

# Prediction on the test set
y_pred_mlr = mlr.predict(X_test)

# Print model coefficients and intercept
print("Intercept:", mlr.intercept_)
print("Coefficients:")
print(list(zip(X.columns, mlr.coef_)))

# Calculate evaluation metrics
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(meanSqErr)

print('Root Mean Square Error:', rootMeanSqErr)
print('Mean Absolute Error:', meanAbErr)
print('R-squared:', mlr.score(X_test, y_test))

###1
plt.plot(np.arange(len(y_test)), y_test, color='blue', label='Actual attendance')
plt.plot(np.arange(len(y_test)), y_pred_mlr, color='red', label='Predicted attendance')
plt.xlabel('Week')
plt.ylabel('Attendance')
plt.title('Actual attendance vs Predicted attendance ')
plt.grid(True)
plt.legend()
plt.show()

##2
plt.plot(dataset['Week'], train_data['Module_X_Lecture_attendance'], color='red',  label='Actual Module Lecture attendance')
plt.plot(dataset['Week'], train_data['Campus_Lecture_attendance'], color='blue',  label='Actual on Campus attendance on Lecture day')

plt.title('Week VS (predicting) Module_X_Lecture_attendance', )
plt.xlabel('Week', )
plt.ylabel('Module_X_Lecture_attendance', )
plt.title('Multiple Linear MLR: On Campus Attendance vs Actual attendance')
plt.grid(True)
plt.legend()
plt.show()


###3

plt.plot(dataset['Week'], train_data['Module_X_Lecture_attendance'], color='green',  label='Actual Lecture attendance')
plt.plot(dataset['Week'], train_data['Campus_Lecture_attendance'], color='blue',  label='Actual Campus attendance on Lecture day')
plt.plot(np.arange(len(y_test)), y_pred_mlr, color='red', label='Predicted attendance on Lecture day')
plt.legend()
#
plt.title('Actual VS Predicted VS on Campus attendance',)
plt.xlabel('Week',)
plt.ylabel('Module_X_Lecture_attendance',)
plt.title('Multiple Linear MLR: On Campus Attendance vs Actual attendance')
plt.grid(True)
plt.legend()
plt.show()
#plt.box

# Scatter plots to visualize the relationships
fig, axs = plt.subplots(2, 1, figsize=(8, 5))
axs = axs.flatten()

dependent_attrs = ['Week', 'Campus_Lecture_attendance', ]

for i, attr in enumerate(dependent_attrs):
    axs[i].scatter(dataset[attr], dataset['Module_X_Lecture_attendance'], color='blue')
    axs[i].set_xlabel(attr)
    axs[i].set_ylabel('Module_X_Lecture_attendance')

plt.tight_layout()
plt.show()