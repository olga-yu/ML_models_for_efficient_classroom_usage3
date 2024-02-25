import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates  # Import the mdates module

# Read the CSV file
data = pd.read_csv('../../dataMSProject/data3.csv')

# Convert the timestamp column to datetime format
data['time'] = pd.to_datetime(data['time'].str[:-1], format='%Y-%m-%dT%H:%M:%S.%f')

# Filter the data for time between 9 am and 6 pm
filtered_data = data[(data['time'].dt.hour >= 9) & (data['time'].dt.hour < 18)]
print(filtered_data)

filtered_data.to_csv('filtered_data4.csv', index=False)

# Extract the input features (X) and the target variable (y) from the test data
humidity_data = filtered_data['humidity']

# Plot the humidity data
plt.figure(figsize=(10, 7))  # Adjust the width and height as needed

plt.plot(filtered_data['time'], humidity_data)
plt.xlabel('Time', fontsize=17)
plt.ylabel('Humidity', fontsize=17)
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
filtered_data = data[(data['time'].dt.hour >= 9) & (data['time'].dt.hour < 18)]
humidity_data = filtered_data['humidity']

# Set the locator and formatter for the x-axis
hours = mdates.HourLocator(interval=30)  # Set the locator to show hours
h_fmt = mdates.DateFormatter('%H:%M')  # Format x-axis to display hours and minutes
plt.gca().xaxis.set_major_locator(hours)
plt.gca().xaxis.set_major_formatter(h_fmt)

x_ticks = filtered_data['time'].iloc[::len(filtered_data)//10]  # Adjust the denominator for desired number of ticks
plt.xticks(x_ticks, rotation=45)

# x_ticks = filtered_data[filtered_data['time'].dt.minute == 45]['time']  # Selecting hours with minute equal to 0
# plt.xticks(x_ticks, x_ticks.dt.strftime('%H:%M'), rotation=45)

plt.show()
