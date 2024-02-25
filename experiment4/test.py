#updated time column
import pandas as pd
from matplotlib import pyplot as plt

# Read the CSV file
data = pd.read_csv('dataMSProject/data.csv')

# Convert the timestamp column to datetime format
data['time'] = pd.to_datetime(data['time'].str[:-1], format='%Y-%m-%dT%H:%M:%S.%f')



# Extract the time component and set it as the index
data['time'] = data['time'].dt.strftime('%H:%M')  # Extract and format hours and minutes
data.set_index('time', inplace=True)

# Save the modified data to a new CSV file
data.to_csv('dataMSProject/modified_file.csv')

# Read the modified data from the new CSV file
df = pd.read_csv('dataMSProject/modified_file.csv')

# Convert the 'time' column to datetime format
df['time'] = pd.to_datetime(df['time'])

df = df.sort_values(by='time')
# Extract the input features (X) and the target variable (y) from the test data
temperature_data = df['temperature']

# Plot the temperature data
plt.plot(df['time'], temperature_data)
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Temperature over Time')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.show()
