import pandas as pd
m,m,
clean_data = pd.read_csv('Humidity-data_Motion_Temperature_9-13.csv')
clean_data2 = pd.read_csv('Humidity-data_Motion_Temperature_14-18.csv')

clean_data = clean_data.dropna()
clean_data2 = clean_data2.dropna()

print(clean_data.head())


clean_data.to_csv('output_9_13.csv',index=False)
clean_data2.to_csv('output_14_18.csv',index=False)