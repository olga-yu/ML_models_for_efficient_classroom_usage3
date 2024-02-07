# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# #from tensorflow.keras.models import load_model
#
#
# from tensorflow.python.keras.models import load_model
#
# # Load the test data from the CSV file
#
# df = pd.read_csv(r'..\experiment4\dataMSProject\data.csv', parse_dates=['time'])
# # Extract the input features (X) and the target variable (y) from the test data
#
# #df.index = df['pressure']
# #df.drop(columns='pressure', inplace=True)
#
# # print(df.head(5))
# #
# # pressure_data = df['temperature']
# # print(df.head(pressure_data.head(5)))
#
#
# print(df.head(5))
#
# pressure_data = df['temperature']
# print(pressure_data.head(5))
#
# df.plot(x='x', y='y', kind='line')
# plt.show()
#
# #pressure_data.plot(kind='bar')
import matplotlib.pyplot as plt
#
# # Create a DataFrame
# #data = {'x': [1, 2, 3, 4, 5], 'y': [2, 3, 5, 7, 11]}
# #df = pd.DataFrame(data)
#
# # Plot the data

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Load the test data from the CSV file
# df = pd.read_csv(r'..\experiment4\dataMSProject\data.csv', index_col=0, parse_dates=['time'])
#
# # Extract the input features (X) and the target variable (y) from the test data
# temperature_data = df['temperature']
#
# # Plot the temperature data
# temperature_data.plot(kind='line', x=df.index)
# plt.xlabel('Time')
# plt.ylabel('Temperature')
# plt.title('Temperature over Time')
# plt.show()
# #previous
#
# from datetime import datetime
#
# time_string = "2024-01-11T02:24:03.890023521Z"
# parsed_time = datetime.fromisoformat(time_string.replace('Z', '+00:00'))
# print(parsed_time)
import pandas as pd

# Assuming your CSV file has a column named 'timestamp' containing the time data
data = pd.read_csv('dataMSProject/data.csv')

# Convert the timestamp column to datetime format
#data['time'] = pd.to_datetime(data['time'])

# Extract the time component


################################

from datetime import datetime

# timestamp = "2024-01-11T02:24:03.890023521Z"
# datetime_obj = datetime.fromisoformat(timestamp[:-1])  # Removing the 'Z' at the end
# hours_minutes = datetime_obj.strftime('%H:%M')
# print(hours_minutes)

datetime_obj = datetime.fromisoformat(data[:-1])
hours_minutes = datetime_obj.strftime('%H:%M')

#######################################

#data['time'] = data['time'].dt.time

# Save the modified data to a new CSV file
data.to_csv('dataMSProject/modified_file.csv', index=False)

df = pd.read_csv(r'..\experiment4\dataMSProject\modified_file.csv', index_col=0, parse_dates=['time'])
#
# # Extract the input features (X) and the target variable (y) from the test data
temperature_data = df['temperature']

# Plot the temperature data
temperature_data.plot(kind='line', x=df.index)
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Temperature over Time')
plt.show()
