import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    # Read the CSV file
    data = pd.read_csv('data.csv')

    # Convert the timestamp column to datetime format
    data['time'] = pd.to_datetime(data['time'])

    # Filter the data for time between 9 am and 6 pm
    filtered_data = data[(data['time'].dt.hour >= 9) & (data['time'].dt.hour < 18)]

    # Plot the temperature data
    plt.figure(figsize=(12, 7))  # Adjust the width and height as needed
    plt.plot(filtered_data['time'], filtered_data['temperature'], linestyle='--', color='blue')
    plt.xlabel('Time', fontsize=24, color='red')
    plt.ylabel('Temperature (°C)', fontsize=24, color='red')
    plt.title('Temperature-Time Plot', fontsize=20)
    plt.grid(True)
    plt.xticks(rotation=45, fontsize=9)  # Rotate x-axis labels for better visibility
    plt.yticks(fontsize=16)


    # Set the locator and formatter for the x-axis
    hours = mdates.HourLocator(interval=1)  # Set the locator to show hours
    h_fmt = mdates.DateFormatter('%H:%M')  # Format x-axis to display hours and minutes
    plt.gca().xaxis.set_major_locator(hours)
    plt.gca().xaxis.set_major_formatter(h_fmt)
    # Select x-ticks from the filtered data
    x_ticks = filtered_data.iloc[::len(filtered_data) // 10]['time']
    plt.xticks(x_ticks, rotation=45)

    plt.show()

except FileNotFoundError:
    print("The specified file could not be found. Please check the file path.")
except Exception as e:
    print("An error occurred:", e)
