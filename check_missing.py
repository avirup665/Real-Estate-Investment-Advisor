import pandas as pd

df = pd.read_csv("data/cleaned_data.csv")

print("===== Missing Values =====")
print(df.isnull().sum())

print("\n===== Columns =====")
print(df.columns.tolist())

print("\n===== Data Types =====")
print(df.dtypes)