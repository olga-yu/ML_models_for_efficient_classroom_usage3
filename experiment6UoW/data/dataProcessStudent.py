import pandas as pd
df = pd.read_csv('../data/processed_motionData2025_2.csv', header=0, index_col=0)

df["StudentID"] = df["StudentID"].str.replace("S", "", regex=False)
print(df)
df.to_csv("../data/processed_motionData2025_3.csv")  # output to csv