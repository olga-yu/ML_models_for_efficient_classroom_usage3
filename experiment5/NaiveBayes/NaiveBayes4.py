import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv('../data/output_9_18.csv')

# Process the 'Extracted_Time' column to extract useful features
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

# Drop unnecessary columns (e.g., 'Time' and 'Extracted_Time' after processing)
df = df.drop(columns=['Time', 'Extracted_Time'])

# Encode 'PartOfDay' as numerical categories for the Naive Bayes model
df['PartOfDay'] = df['PartOfDay'].astype('category').cat.codes

# Split into features (X) and target (y)
X = df.drop(columns=['sensor_mo.mean'])
y = df['sensor_mo.mean']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize Naive Bayes classifier
nb = CategoricalNB()

# Train the model
nb.fit(X_train, y_train)

# Make predictions
y_pred = nb.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
