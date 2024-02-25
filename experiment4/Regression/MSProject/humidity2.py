import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates  # Import the mdates module

# Read the CSV file
data = pd.read_csv('../../dataMSProject/data.csv')

# Convert the timestamp column to datetime format
data['time'] = pd.to_datetime(data['time'].str[:-1], format='%Y-%m-%dT%H:%M:%S.%f')

# Filter the data for time between 9 am and 6 pm
filtered_data = data[(data['time'].dt.hour >= 9) & (data['time'].dt.hour < 18)]

# Extract the input features (X) and the target variable (y) from the test data
humidity_data = filtered_data['humidity']

# Plot the humidity data
plt.plot(filtered_data['time'], humidity_data)
plt.xlabel('Time')
plt.ylabel('Humidity')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

# Set the locator and formatter for the x-axis
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Set the major locator to show hours
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Format x-axis to display hours and minutes

plt.show()
