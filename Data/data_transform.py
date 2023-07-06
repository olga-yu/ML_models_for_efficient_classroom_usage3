from sklearn.preprocessing import LabelEncoder
import pandas as pd
from category_encoders.ordinal import OrdinalEncoder
import category_encoders
#from category_encoders import OrdinalEncoder
#from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler

'''
1. Data transform takes place to transform original textual data into numbered data
- the data is saved to result_output.csv
- data preprocessing  with data2017.csv
-- attendance is sorted against possible bigger and larger rooms
-- attendance trend is  reviewed closer to the end of semester
- comparison is made to original lecture and tutorial sorting and possible bigger and smaller lecture
# bigger and smaller size tutorial size rooms instead
2. CAP_NN.py - contains neural network that should produce predicted attendance, but produces. [[1.]]
3. CAP_NN_testing - test against  new data result_output.csv, compare results MAE, MSE.
- Compare predictions against actual results, but it predictions do not exist
- produce graphs
'''

le = LabelEncoder()
# for data 1 Real Attendance data is present

# Training data
# real data
df = pd.read_csv('data2017.csv', parse_dates=['date'])
df.fillna(0, inplace=True)
print(df.shape)


# Encode the categorical variables
df['room_name'] = le.fit_transform(df['room_name'])
df['week'] = le.fit_transform(df['week'])
df['day'] = le.fit_transform(df['day'])
df['time_of_day'] = le.fit_transform(df['time_of_day'])
df['class_type'] = le.fit_transform(df['class_type'])
df['faculty'] = le.fit_transform(df['faculty'])
df['school'] = le.fit_transform(df['school'])
df['joint'] = le.fit_transform(df['joint'])
df['status'] = le.fit_transform(df['status'])
df['degree'] = le.fit_transform(df['degree'])
#
# def normalize_attendance(df, maxOccupancy, enrollmentCount, operation):
#     return df.apply(lambda row: operation(row[maxOccupancy]/row[enrollmentCount]), axis=1)


#df['normalized_attendance'] = df['normalized_attendance'].normalize_attendance(df, 'attendance', 'enrollment', lambda x, y: x / y )

#df["normalized_attendance"] = df["normalized_attendance"].apply(lambda x, y: normalize_attendance(df, df['attendance'], df['enrollment']), lambda x, y: x / y )

#################################
# Ordinal encodin
# g for duration
#################################
maplist = [
    {
        'col': 'class_duration',
        'mapping': {
            '1:00:00': 1,
            '1:30:00': 1.5,
            '2:00:00': 2,
            '3:00:00': 3,
            '4:00:00': 4,
        }
    }
]

oe = OrdinalEncoder(mapping=maplist)
df = oe.fit_transform(df)
df.drop(['year', 'time_of_day', 'start_time', 'end_time'], axis=1)
df = df.drop(['Unnamed: 0', 'Unnamed: 0', 'Unnamed: 0', 'semester', 'start_time', 'end_time', ], axis=1)
df.info()

stats = df.describe()
print('stats', stats)
###################################
#to replace the class_type_new
##################################
vals_to_replace = {'class1': '1', 'class2': '2'}
df['attendance_by_class'] = df['attendance_by_class'].map(vals_to_replace)
df['attendance_by_class'] = df.attendance_by_class.astype('int64')

vals_to_replace = {'class1': '1', 'class2': '2', 'class3': '3', 'class4': '4', 'class5': '5'}
df['class_type_new'] = df['class_type_new'].map(vals_to_replace)
df['class_type_new'] = df.class_type_new.astype('int64')




###################


#new season update#############

le = LabelEncoder()

# # written by Olga
def season(month):
    # season function
    if month in [0, 1, 2]:
        return 0  # winter
    elif month in [3, 4, 5]:
        return 1  # spring
    elif month in [6, 7, 8]:
        return 2  # summer
    elif month in [9, 10, 11]:
        return 3  # winter

def transform_to_zero(num):
    # transforms the value into zero
    return 0

# to predict attendance neet the csv with attendance attendance_by_class
#################################

##########
#df['date'] = pd.to_datetime(df['date'])

df['date-year'] = df['date'].dt.year

df['date-month'] = df['date'].dt.month

df['date-day'] = df['date'].dt.day
#df['date-day'] = pd.DatetimeIndex(df['date']).month

#df['day'] = df['date'].dt.day
scaler = MinMaxScaler()
#time_components = ['year', 'month', 'day']
#df[time_components] = scaler.fit_transform(df[time_components])

###################################
#print(season(pd.DatetimeIndex(df['date']).month))
#to encode column season
#df['date:season'] = le.fit_transform(df['date:season'])





######################################

# Print the encoded data
print(df.head())
df.to_csv("output_lecture_seminar_processed.csv")  # output to csv
