import pandas as pd
from matplotlib import pyplot as plt

# Read the CSV file
data = pd.read_csv('dataMSProject/data.csv')

# Convert the timestamp column to datetime format
#data['time'] = pd.to_datetime(data['time'].str[:-1], format='%Y-%m-%dT%H:%M:%S.%f')
data['time'] = pd.to_datetime(data['time'].str[:-1], format='T%H:%M')

import pandas as pd

df = pd.DataFrame(data)

df['time'] = pd.to_datetime(df['time'].str[:-1], format='%Y-%m-%dT%H:%M:%S.%f')

# Extracting only hour and minute
df['hour_minute'] = df['time'].dt.strftime('%H:%M')



# Extracting only hour and minute
df['hour_minute'] = df['time'].dt.strftime('%H:%M')


# Extract the input features (X) and the target variable (y) from the test data
temperature_data = data['temperature']

# Plot the temperature data
plt.plot(data['time'], temperature_data)
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Temperature over Time')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.show()
