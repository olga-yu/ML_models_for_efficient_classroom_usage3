#MOTION detection paper
import os
import pandas as pd
import matplotlib.pyplot as plt
# Construct the file path for motion data
file_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'Motion-data 9-13.csv')
# Read the CSV file
try:
    data = pd.read_csv(file_path)
    print("File successfully read:")
    print(data)

    # Convert 'Time' column to datetime format

    data['Time'] = pd.to_datetime(data['Time'], dayfirst=True)

    # Extract hour from 'Time' column
    data['Hour'] = data['Time'].dt.strftime('%H:%M')
    # Drop rows with missing values
    data.dropna(subset=['Time', 'sensor_mo.mean'], inplace=True)  # Adjust column name to 'motion'
    # Create the plot
    plt.figure(figsize=(12, 6))  # Set figure size
    plt.plot(data['Hour'], data['sensor_mo.mean'], linestyle='--', color='blue')  # Adjust column name to 'motion'
    plt.xlabel('Time', fontsize=24, color='red')  # Set label color to red
    plt.ylabel('Motion', fontsize=24, color='red')  # Change ylabel to motion
    plt.title('Motion-Time Plot, 9:00 - 13:00, Student A', fontsize=20)  # Change title
    plt.grid(True)
    plt.xticks(rotation=45, fontsize=9)  # Rotate the hour labels by 45 degrees and reduce font size
    plt.yticks(fontsize=16)  # Reduce font size of y-axis labels

    # Adjust y-axis ticks
    plt.tight_layout()  # Adjust layout to prevent clipping of x-axis labels
    plt.show()
except FileNotFoundError:
    print("The specified file could not be found. Please check the file path.")
except Exception as e:
    print("An error occurred:", e)
