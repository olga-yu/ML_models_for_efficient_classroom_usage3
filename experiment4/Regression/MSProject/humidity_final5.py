import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates  # Import the mdates module

# Read the CSV file
data = pd.read_csv('data.csv')

# Convert the timestamp column to datetime format
data['time'] = pd.to_datetime(data['time'].str[:-1], format='%Y-%m-%dT%H:%M:%S.%f')

# Filter the data for time between 9 am and 6 pm
filtered_data = data[(data['time'].dt.hour >= 9) & (data['time'].dt.hour < 18)]

# Extract the input features (X) and the target variable (y) from the filtered data
humidity_data = filtered_data['humidity']

# Plot the humidity data
plt.figure(figsize=(12, 8))

plt.plot(filtered_data['time'], humidity_data, linestyle='--', color='blue')
plt.xlabel('Time', fontsize=24, color='red')
plt.ylabel('Humidity (%)', fontsize=24, color='red')
plt.title('Humidity-Time Plot', fontsize=20)
plt.grid(True)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)

# Customize y-axis ticks
min_humidity = int(humidity_data.min())  # Find the minimum humidity value
max_humidity = int(humidity_data.max())  # Find the maximum humidity value
y_ticks = [i for i in range(min_humidity, max_humidity + 1)]
y_ticks_with_half = []
for tick in y_ticks:
    y_ticks_with_half.append(tick)
    y_ticks_with_half.append(tick + 0.5)
plt.yticks(y_ticks_with_half)

#x ticks
# Set the locator and formatter for the x-axis

# Extract hour from 'time' column of filtered data
hours = filtered_data['time'].dt.hour.unique()

# Set x-axis ticks to correspond with the hours of the filtered data
plt.xticks(ticks=filtered_data['time'][::len(filtered_data)//len(hours)], labels=hours, rotation=45, fontsize=10)

####

hours = mdates.HourLocator(interval=30)  # Set the locator to show hours
h_fmt = mdates.DateFormatter('%H:%M')  # Format x-axis to display hours and minutes


plt.gca().xaxis.set_major_locator(hours)
plt.gca().xaxis.set_major_formatter(h_fmt)

#####
plt.tight_layout()
plt.show()
