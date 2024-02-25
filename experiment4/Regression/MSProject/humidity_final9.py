import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

try:
    # Read the CSV file
    data = pd.read_csv('../../dataMSProject/data3.csv')

    # Convert the timestamp column to datetime format
    data['time'] = pd.to_datetime(data['time'].str[:-1], format='%Y-%m-%dT%H:%M:%S.%f')

    # Filter the data for time between 9 am and 6 pm
    filtered_data = data[(data['time'].dt.hour >= 9) & (data['time'].dt.hour < 18)]

    # Extract the input features (X) and the target variable (y) from the filtered data
    humidity_data = filtered_data['humidity']

    # Plot the humidity data
    plt.figure(figsize=(10, 7))

    plt.plot(filtered_data['time'], humidity_data)
    plt.xlabel('Time', fontsize=17)
    plt.ylabel('Humidity', fontsize=17)
    plt.xticks(rotation=45)

    # Set the locator and formatter for the x-axis
    hours = mdates.HourLocator(interval=2)  # Set the locator to show every 2 hours
    h_fmt = mdates.DateFormatter('%H:%M')  # Format x-axis to display hours and minutes
    plt.gca().xaxis.set_major_locator(hours)
    plt.gca().xaxis.set_major_formatter(h_fmt)

    plt.tight_layout()
    plt.show()

except Exception as e:
    print("An error occurred:", e)