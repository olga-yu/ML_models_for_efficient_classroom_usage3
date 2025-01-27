import pandas as pd

df = pd.read_csv('../data/updated_timetable_with_attendance_v7.csv')

def process_time(time_str):
    if time_str <=3 :
        return 1
    elif 3 < time_str < 7 :
        return 2
    elif 7 < time_str < 10:
        return 3
    else:
        return 4

#print(process_time(df['partOfDay'] = df['period'].apply(process_time)))

df['partOfDay'] = df['period'].apply(process_time)
print(df)
df.to_csv("processed_timetable_with_attendance_v7.csv")  # output to csv
