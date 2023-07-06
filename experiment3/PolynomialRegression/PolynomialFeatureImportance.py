import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r'../../Data/westminster.csv')


feature_cols = ['Week', 'Reg', 'Campus_Lecture_attendance', 'Module_X_Lecture_attendance', 'Module_A_deadlines', 'Campus_Lecture_attendance',
                'Module_B_deadlines', 'Module_C_deadlines']

X = dataset[feature_cols]
y = dataset['Module_X_Lecture_attendance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

poly_regr = LinearRegression()
poly_regr.fit(X_poly_train, y_train)
y_pred = poly_regr.predict(X_poly_test)


meanAbErr = metrics.mean_absolute_error(y_test, y_pred)
meanSqErr = metrics.mean_squared_error(y_test, y_pred)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r_squared = poly_regr.score(X_poly_test, y_test)

print('Root Mean Square Error:', rootMeanSqErr)
print('Mean Absolute Error:', meanAbErr)
print('R squared: {:.2f}'.format(r_squared * 100))
print('Mean Square Error:', meanSqErr)

# Plot the feature importances
importance = abs(poly_regr.coef_)
feature_names = poly.get_feature_names_out(feature_cols)
sorted_idx = importance.argsort()[::-1]

plt.barh(range(X_poly_train.shape[1]), importance[sorted_idx], tick_label=np.array(feature_names)[sorted_idx])
plt.title('Feature Importances')
plt.xlabel('Relative Importance')
plt.show()
