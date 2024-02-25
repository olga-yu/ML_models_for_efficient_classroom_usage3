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
filtered_data.to_csv('filtered_data.csv', index=False)

# Extract the input features (X) and the target variable (y) from the test data
temperature_data = filtered_data['temperature']

# Plot the humidity data
plt.plot(filtered_data['time'], temperature_data)
plt.xlabel('Time', fontsize=17)
plt.ylabel('Temperature', fontsize=17)
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
filtered_data = data[(data['time'].dt.hour >= 9) & (data['time'].dt.hour < 18)]
temperature_data = filtered_data['temperature']

# Set the locator and formatter for the x-axis
hours = mdates.HourLocator(interval=30)  # Set the locator to show hours
h_fmt = mdates.DateFormatter('%H:%M')  # Format x-axis to display hours and minutes
plt.gca().xaxis.set_major_locator(hours)
plt.gca().xaxis.set_major_formatter(h_fmt)

plt.show()
