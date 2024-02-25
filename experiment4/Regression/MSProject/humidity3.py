import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates  # Import the mdates module

# Read the CSV file
data = pd.read_csv('../../dataMSProject/data.csv')

# Convert the timestamp column to datetime format
data['time'] = pd.to_datetime(data['time'].str[:-1], format='%Y-%m-%dT%H:%M:%S.%f')

# Extract the input features (X) and the target variable (y) from the test data
temperature_data = data['humidity']

# Plot the temperature data
plt.plot(data['time'], temperature_data)
plt.xlabel('Time')
plt.ylabel('Humidity')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

# Set the locator and formatter for the x-axis
filtered_data = data[(data['time'].dt.hour >= 9) & (data['time'].dt.hour < 18)]
print(filtered_data)
# Extract the input features (X) and the target variable (y) from the test data
humidity_data = filtered_data['humidity']
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=30))  # Set the major locator to show hours
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Format x-axis to display hours and minutes

plt.show()
