import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates  # Import the mdates module

# Read the CSV file
data = pd.read_csv('../../dataMSProject/data.csv')

# Convert the timestamp column to datetime format
data['time'] = pd.to_datetime(data['time'].str[:-1], format='%Y-%m-%dT%H:%M:%S.%f')



# Filter the data for time between 9 am and 6 pm
filtered_data = data[(data['time'].dt.hour >= 9) & (data['time'].dt.hour < 18)]
print(filtered_data)

filtered_data.to_csv('humidity_filtered_data.csv', index=False)

# Convert 'Time' column to datetime format
filtered_data['Time'] = pd.to_datetime(filtered_data['Time'], dayfirst=True)

    # Extract hour from 'Time' column
data['Hour'] = data['Time'].dt.strftime('%H:%M')
# Extract the input features (X) and the target variable (y) from the test data
humidity_data = filtered_data['humidity']

# Plot the humidity data
plt.figure(figsize=(12, 6))  # Adjust the width and height as needed

plt.plot(data['Hour'], data['sensor_hu.mean'], linestyle='--', color='blue')  # Set line style to dashed
plt.xlabel('Time', fontsize=24, color='red')  # Set label color to red
plt.ylabel('Humidity (%)', fontsize=24, color='red')  # Change ylabel to humidity
plt.title('Humidity-Time Plot', fontsize=20)  # Change title
plt.grid(True)
plt.xticks(rotation=45, fontsize=9)  # Rotate the hour labels by 45 degrees and reduce font size
plt.yticks(fontsize=16)  # Reduce font size of y-axis labels

# Set the locator and formatter for the x-axis
hours = mdates.HourLocator(interval=30)  # Set the locator to show hours
h_fmt = mdates.DateFormatter('%H:%M')  # Format x-axis to display hours and minutes
plt.gca().xaxis.set_major_locator(hours)
plt.gca().xaxis.set_major_formatter(h_fmt)

# x_ticks = filtered_data[filtered_data['time'].dt.minute == 45]['time']  # Selecting hours with minute equal to 0
# plt.xticks(x_ticks, x_ticks.dt.strftime('%H:%M'), rotation=45)

plt.show()
