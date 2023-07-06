import pandas as pd
import numpy as np
#from tensorflow.keras.models import load_model

from tensorflow.python.keras.models import load_model


# Load the test data from the CSV file

df = pd.read_csv(r'..\Data\output_lecture_seminar_processed3.csv', parse_dates=['date'])
# Extract the input features (X) and the target variable (y) from the test data
X_test = df.drop('attendance', axis=1).values
y_test = df['attendance'].values

# Load your previously trained neural network
model = load_model('./CAP_NN_paper.py')

# Use the trained model to make predictions on the test set
y_pred = model.predict(X_test)

# Compare the predicted values to the actual values
mse = np.mean((y_pred - y_test)**2)
print("Mean Squared Error: {:.2f}".format(mse))
