from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
import pandas as pd

# define dataset
#X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)

X, y = make_classification(n_samples=1000, n_features=11, n_informative=5, n_redundant=6, random_state=1)

dataset = pd.read_csv(r'../june2024/Humidity-data9-13.csv', parse_dates=['date'])
# data cleaning

#

# Select the relevant features
feature_cols = ['Time', 'sensor_hu.mean']

# Extract the features and target variable
X = dataset[feature_cols]
y = dataset['normalized_attendance2']

# define the model

model = DecisionTreeClassifier()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance

for i,v in enumerate(importance):
 print( 'Feature: %0d, Score: %.5f' % (i,v))

for feature, score in zip(feature_cols, importance):
    print(feature, score)

# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
pyplot.xlabel()
