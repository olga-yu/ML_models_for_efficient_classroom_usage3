import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

# Read the CSV file
data = pd.read_csv('data1.csv')

# Convert the timestamp column to datetime format
data['time'] = pd.to_datetime(data['time'])

# Filter the data for time between 9 am and 6 pm
filtered_data = data[(data['time'].dt.hour >= 9) & (data['time'].dt.hour < 18)]

# Extract hour from 'time' column
filtered_data['Hour'] = filtered_data['time'].dt.strftime('%H:%M')

# Plot the humidity data
plt.figure(figsize=(12, 6))

plt.plot(filtered_data['Hour'], filtered_data['sensor_hu.mean'], linestyle='--', color='blue')
plt.xlabel('Time', fontsize=24, color='red')
plt.ylabel('Humidity (%)', fontsize=24, color='red')
plt.title('Humidity-Time Plot', fontsize=20)
plt.grid(True)
plt.xticks(rotation=45, fontsize=9)

# Set the locator and formatter for the x-axis
hours = mdates.HourLocator(interval=1)  # Show hours with 1-hour interval
h_fmt = mdates.DateFormatter('%H:%M')
plt.gca().xaxis.set_major_locator(hours)
plt.gca().xaxis.set_major_formatter(h_fmt)

plt.tight_layout()
plt.show()
