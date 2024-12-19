import pandas as pd

# Read the CSV file
df = pd.read_csv('../data/output_9_18.csv', header=0, index_col=0)

# Convert 'Extracted_Time' to datetime format, assuming the times are in '%H:%M' format
df['Extracted_Time'] = pd.to_datetime(df['Extracted_Time'], format='%H:%M').dt.time

# Extract hour and minute features (if needed for analysis)
df['Hour'] = df['Extracted_Time'].apply(lambda x: x.hour)
df['Minute'] = df['Extracted_Time'].apply(lambda x: x.minute)

# Convert the time into minutes past midnight for time-series models:
df['Minutes_Past_Midnight'] = df['Hour'] * 60 + df['Minute']

# Display the transformed dataframe
print(df)
df.to_csv("output_9_18_minutes.csv")  # output to csv
