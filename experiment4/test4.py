import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

# Read the CSV file
data = pd.read_csv('dataMSProject/data.csv')

# Convert the timestamp column to datetime format
data['time'] = pd.to_datetime(data['time'].str[:-1], format='%Y-%m-%dT%H:%M:%S.%f')
# Extracting only hour and minute
df = pd.DataFrame(data)

df['hour_minute'] = df['time'].dt.strftime('%H:%M')

#data['time'] = pd.to_datetime(data['time'])
# Extract the input features (X) and the target variable (y) from the test data
temperature_data = data['temperature']

# Plot the temperature data
plt.plot(data['time'], temperature_data)
plt.xlabel('Time', )
plt.ylabel('Temperature')
plt.title('Temperature over Time')
#plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.xticks(data['time'], data['time'].dt.strftime('%H:%M'), rotation=45)
#plt.gca().xaxis.set_major_locator(plt.MultipleLocator(3000000))    # Display only hours and minutes on x-axis
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=30))  # Set the major locator to show hours

plt.show()
