'''

'''
import csv
from datetime import datetime
import datetime
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import category_encoders
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

le = LabelEncoder()

df = pd.read_csv(r'..\Data\output_lecture_seminar_processed3.csv', parse_dates=['date'])
#################################
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
scaler = MinMaxScaler()
time_components = ['year', 'month', 'day']
df[time_components] = scaler.fit_transform(df[time_components])

###################################
data_input = df.drop(['Unnamed: 0', 'year', 'date','room_name', 'status', 'year','attendance', 'attendance_by_class', 'date-year', 'date-month', 'date-day','normalized_attendance','class_type_new'], axis=1)  #

print("after drop shape and head")
print(data_input.head)
print("Data input")
print(data_input.shape)

print("data_output")
data_output = df[['normalized_attendance']].T
print(data_output.shape)

class NeuralNetwork():
    def __init__(self):
        # seeding for random number generation
        np.random.seed(1)
        # ValueError: shapes (2136,12) and (5,1) not aligned: 12 (dim 1) != 5 (dim 0)
        self.synaptic_weights = 2 * np.random.random((11, 1)) - 1

    def sigmoid(self, x):
        # the sigmoid function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # derivative to the Sigmoid function
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        # training the model to make accurate predictions while adjusting weights continually
        for iteration in range(training_iterations):
            # siphon the training data via  the neuron
            output = self.think(training_inputs)
            # computing error rate for back-propagation
            error = training_outputs - output
            print("error", output)
            # performing weight adjustments
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def think(self, inputs):
        inputs = inputs.astype(int or str)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        return output



if __name__ == "__main__":
    # initializing the neuron class
    neural_network = NeuralNetwork()

    print("Beginning Randomly Generated Weights: (synaptic weights) ")
    print(neural_network.synaptic_weights, len(neural_network.synaptic_weights), neural_network.synaptic_weights.shape)

    print(data_input.shape)
    training_inputs = np.array(data_input)
    training_outputs = np.array(data_output).T
    print("training output")
    # training taking place
    neural_network.train(training_inputs, training_outputs, 1500)
    # user_input_one = str(input("P1: "))
    # user_input_two = str(input("P2: "))
    # user_input_three = str(input("P3: "))
    # user_input_four = str(input("P4: "))
    # user_input_five = str(input("P5: "))
    # user_input_six = str(input("L1: "))
    # user_input_seven = str(input("L2: "))
    # user_input_eight = str(input("A1: "))
    # user_input_nine = str(input("A2: "))
    # user_input_ten = str(input("A3: "))
    # user_input_eleven = str(input("F1: "))
    # user_input_twelve = str(input("F2: "))
    # user_input_thirteen = str(input("F3: "))
    # user_input_fourteen = str(input("F4: "))
    # user_input_fourteen = str(input("SA1: "))
    # user_input_fourteen = str(input("SA2: "))


    with open(r'..\Data\output_lecture_seminar_processed3.csv') as csvfile:  # Testing Input Data
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_type = int(row['class_type'])
            faculty = int(row['faculty'])
            school = int(row['school'])
            enrollment = int(row['enrollment'])
            class_duration = int(row['class_duration'])  # time variable
            degree = int(row['degree'])
            status = int(row['status'])
            joint = int(row['joint'])
            week = int(row['week'])
            day = int(row['day'])
            time_of_day = int(row['time_of_day'])


print("Attributes predicting attendance")

print("Class type(lecture/laboratory/tutorial)*: ", class_type, )
print("Course is combined or with other courses or not*", joint, )
print("Faculty*: ", faculty, )
print("Enrollment*", enrollment, )
print("School* ", school, )
print("Degree* ", degree, )
print("Status* (open or full) ", status, )
print("Time of the day: ", time_of_day, )
print("Day", day, )
print("Week ", week, )
print("Enrollment", enrollment)

final_result = neural_network.think(np.array(
    [class_type, faculty, school, enrollment, class_duration, degree, class_type, joint, week, day,  time_of_day
     ])),
# 11 inputs listed above


final_result = np.round(final_result, 5)
print("Final result: ", final_result)
print("Meaning:\n"
      "0 - no student attended, or class cancellation \n"
      "1 - all enrolled students attended")