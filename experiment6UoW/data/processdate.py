import pandas as pd

# Read the CSV file
df = pd.read_csv('../data/processed_motionData2025_1.csv', parse_dates=['Date'], header=0)

# Convert 'Date' column to datetime format, handling errors
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Extract year, month, day
df['date-year'] = df['Date'].dt.year
df['date-month'] = df['Date'].dt.month
df['date-day'] = df['Date'].dt.day

# Define season function
def season(month):
    if month in [12, 1, 2]:  # Winter
        return 0
    elif month in [3, 4, 5]:  # Spring
        return 1
    elif month in [6, 7, 8]:  # Summer
        return 2
    elif month in [9, 10, 11]:  # Fall (Autumn)
        return 3
    else:
        return None  # Handle NaN or invalid cases

def semester(month):
    if month in [9,10,11,12]:
        return 0 # Semester 1
    elif month in [1,2,3,4,5]:
        return 1 # Semester 2
    elif month in [6,7,8]:
        return 2 # Semester 9 Referral
    else: return None

df['Season'] = df['date-month'].apply(season)
df['Semester'] = df['date-month'].apply(semester)

# Add weekday columns (numeric format)
df['Weekday'] = df['Date'].dt.weekday      # 0 = Monday, 6 = Sunday


# Check for NaT (missing dates)
print("Missing dates:", df['Date'].isna().sum())

print(df.head())

# Save to CSV
df.to_csv("processed_motionData2025_3.csv", index=False)
