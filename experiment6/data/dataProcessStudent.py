import pandas as pd
df = pd.read_csv('../data/motionData2025.csv', header=0, index_col=0)

df["StudentID"] = df["StudentID"].str.replace("S", "", regex=False)
print(df)
df.to_csv("motionData2025_1.csv")  # output to csv