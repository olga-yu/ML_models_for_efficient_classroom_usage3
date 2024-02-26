import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates  # Import the mdates module

try:
    # Read the CSV file
    #data = pd.read_csv('../../dataMSProject/data.csv')

    #data = pd.read_csv('https://raw.githubusercontent.com/olga-yu/ML_models_for_efficient_classroom_usage3/blob/master/experiment4/dataMSProject/data.csv')
    data = pd.read_csv(
        'https://raw.githubusercontent.com/olga-yu/ML_models_for_efficient_classroom_usage3/master/experiment4/dataMSProject/data.csv')

    # Convert the timestamp column to datetime format
    data['time'] = pd.to_datetime(data['time'].str[:-1], format='%Y-%m-%dT%H:%M:%S.%f')

    # Filter the data for time between 9 am and 6 pm
    filtered_data = data[(data['time'].dt.hour >= 9) & (data['time'].dt.hour < 18)]
    print(filtered_data)

    filtered_data.to_csv('humidity_filtered_data.csv', index=False)

    # Extract the input features (X) and the target variable (y) from the test data
    humidity_data = filtered_data['humidity']

    # Drop rows with missing values
    data.dropna(subset=['Time'], inplace=True)

    # Plot the humidity data
    plt.figure(figsize=(12, 6))  # Adjust the width and height as needed

    plt.plot(filtered_data['time'], humidity_data, linestyle='--', color='blue')
    plt.xlabel('Time', fontsize=24, color='red')
    plt.ylabel('Humidity (%)', fontsize=24, color='red')
    plt.title('Humidity-Time Plot', fontsize=20)  # Change title

    plt.xticks(rotation=45, fontsize=9)  # Rotate the hour labels by 45 degrees and reduce font size

    # Redefinition of filtered_data and humidity_data is redundant and can be removed
    # Rotate x-axis labels for better visibility
    # filtered_data = data[(data['time'].dt.hour >= 9) & (data['time'].dt.hour < 18)]
    # humidity_data = filtered_data['humidity']

    plt.yticks(fontsize=16)  # Reduce font size of y-axis labels

    # Customize y-axis ticks
    y_ticks = [i for i in range(int(filtered_data.min()), int(filtered_data.max()) + 1)]
    y_ticks_with_half = []
    for tick in y_ticks:
        y_ticks_with_half.append(tick)
        y_ticks_with_half.append(tick + 0.5)
    plt.yticks(y_ticks_with_half)

    plt.tight_layout()  # Adjust layout to prevent clipping of x-axis labels
    plt.show()


except FileNotFoundError:
    print("The specified file could not be found. Please check the file path.")
except Exception as e:
    print("An error occurred:", e)