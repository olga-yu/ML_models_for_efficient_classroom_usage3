import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv('../data/augmented_dataset.csv')

# Process 'Extracted_Time' to extract useful features
def process_time(time_str):
    hour = int(time_str.split(':')[0])
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

df['PartOfDay'] = df['Extracted_Time'].apply(process_time)

# Drop unnecessary columns
df = df.drop(columns=['Time', 'Extracted_Time'])

# Encode 'PartOfDay' as numerical categories
df['PartOfDay'] = df['PartOfDay'].astype('category').cat.codes

# Define features (X) and target (y)
X = df.drop(columns=['sensor_mo.mean'])  # Features
y = df['sensor_mo.mean']  # Target (Binary: 0 or 1)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# **Apply SMOTE to handle class imbalance**
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train a Naive Bayes model
nb = GaussianNB()
nb.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred = nb.predict(X_test)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
