import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file from the URL into a pandas DataFrame
url = 'https://raw.githubusercontent.com/wiut-tutor2022/Machine_Learning_Models_for_Efficient_Classroom_Usage2/master/Data/output_lecture_seminar_processed4.csv'
data = pd.read_csv(url)

# Explore the data and identify columns that may contain outliers
print(data.describe())
data.boxplot(column=[ 'attendance', 'enrollment',])
plt.show()
#
# # Identify outliers using statistical methods
z_scores = np.abs((data[''] - data['enrollment'].mean()) / data['attendance'].std())
threshold = 3
outliers = data[z_scores > threshold]
#
# # Visualize the identified outliers
plt.scatter(data['attendance'], data['enrollment'])
plt.scatter(outliers['attendance'], outliers['enrollment'], color='r')
plt.show()