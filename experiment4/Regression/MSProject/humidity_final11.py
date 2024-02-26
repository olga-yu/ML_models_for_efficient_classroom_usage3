import pandas as pd
from matplotlib import pyplot as plt

try:
    # Read the CSV file
    data = pd.read_csv('https://raw.githubusercontent.com/olga-yu/ML_models_for_efficient_classroom_usage3/master/experiment4/dataMSProject/data.csv')

    # Convert the timestamp column to datetime format
    data['time'] = pd.to_datetime(data['time'], dayfirst=True)

    # Filter the data for time between 9 am and 6 pm
    filtered_data = data[(data['time'].dt.hour >= 9) & (data['time'].dt.hour < 18)]
    print(filtered_data)

    data['Hour'] = data['time'].dt.strftime('%H:%M')

    filtered_data.to_csv('humidity_filtered_data.csv', index=False)

    # Extract the input features (X) and the target variable (y) from the test data
    humidity_data = filtered_data['humidity']

    # Drop rows with missing values based on the 'time' column
    filtered_data.dropna(subset=['time', 'sensor_hu.mean'], inplace=True)

    # Check the columns of filtered_data
    print(filtered_data.columns)

    # Plot the humidity data
    plt.figure(figsize=(12, 6))  # Adjust the width and height as needed
    plt.plot(filtered_data['Hour'], filtered_data['sensor_hu.mean'], linestyle='--', color='blue')  # Use filtered_data here

    plt.xlabel('Time', fontsize=24, color='red')
    plt.ylabel('Humidity (%)', fontsize=24, color='red')
    plt.title('Humidity-Time Plot', fontsize=20)

    plt.xticks(rotation=45, fontsize=9)
    plt.yticks(fontsize=16)

    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print("The specified file could not be found. Please check the file path.")
except Exception as e:
    print("An error occurred:", e)
