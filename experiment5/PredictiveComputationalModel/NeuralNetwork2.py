'''

'''
import csv
from datetime import datetime
import datetime
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
#import category_encoders
#import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

le = LabelEncoder()

df = pd.read_csv(r'../data/output_9_18_minutes.csv')
#################################
# df['Time'] = pd.to_datetime(df['Time'])
# df['Month'] = df['ate'].dt.month
# df['Year'] = df['date'].dt.year
# df['day'] = df['date'].dt.day
# scaler = MinMaxScaler()
# time_components = ['year', 'month', 'day']
# df[time_components] = scaler.fit_transform(df[time_components])

###################################
data_input = df.drop(['Time', 'sensor_mo.mean','Minute', 'Minutes_Past_Midnight', 'Extracted_Time'], axis=1)  #

print("after drop shape and head")
print(data_input.head)
print("Data input")
print(data_input.shape)

print("data_output")
data_output = df[['sensor_mo.mean']].T
print(data_output.shape)
class NeuralNetwork():
    def __init__(self):
        # Seed for reproducibility
        np.random.seed(1)
        # Adjust weights to match 12 input features
        self.synaptic_weights = 2 * np.random.random((12, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            # Forward pass
            output = self.think(training_inputs)
            # Compute error
            error = training_outputs - output
            # Weight adjustments
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def think(self, inputs):
        inputs = inputs.astype(float)  # Ensure inputs are numeric
        return self.sigmoid(np.dot(inputs, self.synaptic_weights))

# Main program
if __name__ == "__main__":
    neural_network = NeuralNetwork()

    print("Initial Weights:")
    print(neural_network.synaptic_weights)

    # Ensure data dimensions align
    training_inputs = np.array(data_input)
    training_outputs = np.array(data_output).T  # Transpose for alignment
    print("Training Input Shape:", training_inputs.shape)
    print("Training Output Shape:", training_outputs.shape)

    # Train the model

    with open(r'../data/output_9_18_minutes.csv') as csvfile:  # Testing Input Data
        reader = csv.DictReader(csvfile)
        for row in reader:

            month = int(row['Month'])
            week = int(row['Year'])
            weekDay = int(row['WeekDay'])
            time_of_day = int(row['TimeOfDay'])
            semester = int(row['Semester'])
            hour = int(row['Hour'])



    neural_network.train(training_inputs, training_outputs, 150)

    # Test input
    test_input = np.array([month, week, time_of_day, weekDay, semester, hour])
    print("Test Input Shape:", test_input.shape)

    # Predict
    final_result = neural_network.think(test_input)
    final_result = np.round(final_result, 5)
    print("Final result:", final_result)
