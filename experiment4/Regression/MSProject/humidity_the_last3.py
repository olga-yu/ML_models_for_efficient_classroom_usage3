import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Read the CSV file
data = pd.read_csv('https://raw.githubusercontent.com/olga-yu/ML_models_for_efficient_classroom_usage3/master/experiment4/dataMSProject/data.csv')

# Convert the timestamp column to datetime format
data['time'] = pd.to_datetime(data['time'])

# Filter the data for time between 9 am and 6 pm
filtered_data = data[(data['time'].dt.hour >= 9) & (data['time'].dt.hour < 18)]

# Determine the number of data points
num_points = len(filtered_data)

# Calculate the step for selecting x-ticks
step = num_points // 10  # Select 10 ticks

# Select x-ticks from the filtered data
x_ticks = filtered_data.iloc[::step]['time']

# Plot the desired data
plt.figure(figsize=(10, 6))
plt.plot(filtered_data['time'], filtered_data['humidity'])
plt.xlabel('Time', fontsize=14)
plt.ylabel('Humidity', fontsize=14)
plt.title('Humidity Data between 9 AM and 6 PM', fontsize=16)

# Set the locator and formatter for the x-axis
plt.xticks(x_ticks, rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
