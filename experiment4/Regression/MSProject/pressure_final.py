import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates  # Import the mdates module

# Read the CSV file
data = pd.read_csv('../../dataMSProject/data.csv')

# Convert the timestamp column to datetime format
data['time'] = pd.to_datetime(data['time'].str[:-1], format='%Y-%m-%dT%H:%M:%S.%f')

# Filter the data for time between 9 am and 6 pm
filtered_data = data[(data['time'].dt.hour >= 9) & (data['time'].dt.hour < 18)]

# Remove values above 1030
filtered_data = filtered_data[filtered_data['pressure'] <= 1030]

# Extract the input features (X) and the target variable (y) from the test data
pressure_data = filtered_data['pressure']

# Plot the humidity data
plt.figure(figsize=(10, 7))  # Adjust the width and height as needed

plt.plot(filtered_data['time'], pressure_data)
plt.xlabel('Time', fontsize=24, color='red')
plt.ylabel('Pressure', fontsize=24, color='red')
plt.title('Pressure-Time Plot', fontsize=20)
plt.grid(True)
plt.xticks(rotation=45, fontsize=9)  # Rotate the hour labels by 45 degrees and reduce font size
plt.yticks(fontsize=16)  # Reduce font size of y-axis labels

filtered_data = data[(data['time'].dt.hour >= 9) & (data['time'].dt.hour < 18)]
print(filtered_data)
filtered_data.to_csv('pressure_filtered_data.csv', index=False)

temperature_data = filtered_data['pressure']

# Customize y-axis ticks
y_ticks = [i for i in range(int(data['sensor_hu.mean'].min()), int(data['sensor_hu.mean'].max()) + 1)]
y_ticks_with_half = []
for tick in y_ticks:
    y_ticks_with_half.append(tick)
    y_ticks_with_half.append(tick + 0.5)
plt.yticks(y_ticks_with_half)

plt.tight_layout()  # Adjust layout to prevent clipping of x-axis labels
plt.show()

# Set the locator and formatter for the x-axis
# hours = mdates.HourLocator(interval=30)  # Set the locator to show hours
# h_fmt = mdates.DateFormatter('%H:%M')  # Format x-axis to display hours and minutes
# plt.gca().xaxis.set_major_locator(hours)
# plt.gca().xaxis.set_major_formatter(h_fmt)
#
# x_ticks = filtered_data['time'].iloc[::len(filtered_data)//10]  # Adjust the denominator for desired number of ticks
# plt.xticks(x_ticks, rotation=45)
# plt.show()
