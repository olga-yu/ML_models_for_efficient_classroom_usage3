import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

# Read the CSV file from the URL
data = pd.read_csv('https://raw.githubusercontent.com/olga-yu/ML_models_for_efficient_classroom_usage3/master/experiment4/dataMSProject/data.csv')

# Convert the timestamp column to datetime format
data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%dT%H:%M:%S.%f')

# Filter the data for time between 9 am and 6 pm
filtered_data = data[(data['time'].dt.hour >= 9) & (data['time'].dt.hour < 18)]

# Extract the input features (X) and the target variable (y) from the filtered data
humidity_data = filtered_data['humidity']

# Plot the humidity data
plt.figure(figsize=(10, 7))

plt.plot(filtered_data['time'], humidity_data)
plt.xlabel('Time', fontsize=17)
plt.ylabel('Humidity', fontsize=17)

# Set the locator and formatter for the x-axis
#hours = mdates.HourLocator(interval=5)  # Set the locator to show hours
#h_fmt = mdates.DateFormatter('%H:%M')  # Format x-axis to display hours and minutes

x_ticks = filtered_data['time'].iloc[::len(filtered_data)//10]  # Adjust the denominator for desired number of ticks


plt.xticks(rotation=45)

plt.show()
