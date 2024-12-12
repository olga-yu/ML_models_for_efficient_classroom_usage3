#MOTION detection CIHAN paper
import os
import pandas as pd
import matplotlib.pyplot as plt

# Humidity graph 9_13
# Construct the file path
# file_path = os.path.join(os.path.expanduser('~experiment4/dataMSProject/experiment4/dataMSProject/IFIP/Humidity-data 9-13.csv'), '', 'Humidity-data 9-13.csv')
file_path = os.path.join(
    os.path.expanduser('~experiment4/dataMSProject/experiment4/dataMSProject/IFIP/Humidity-data 9-13.csv'), '',
    'Motion-data 9-13.csv')
# file_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'Humidity-data 9-13.csv')

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
    data.dropna(subset=['Time', 'sensor_hu.mean'], inplace=True)

    # Create the plot
    plt.figure(figsize=(12, 6))  # Set figure size
    plt.plot(data['Hour'], data['sensor_hu.mean'], linestyle='--', color='blue')  # Set line style to dashed
    plt.xlabel('Time', fontsize=24, color='red')  # Set label color to red
    plt.ylabel('Motion (%)', fontsize=24, color='red')  # Change ylabel to humidity
    plt.title('Motion-Time Plot from 9:00 to 13:00', fontsize=20)  # Change title
    plt.grid(True)
    plt.xticks(rotation=45, fontsize=9)  # Rotate the hour labels by 45 degrees and reduce font size
    plt.yticks(fontsize=16)  # Reduce font size of y-axis labels

    # Customize y-axis ticks
    y_ticks = [i for i in range(int(data['sensor_hu.mean'].min()), int(data['sensor_hu.mean'].max()) + 1)]
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
