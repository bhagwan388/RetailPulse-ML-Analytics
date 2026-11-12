import pandas as pd


input_file = "data/processed/daily_sales.csv"
output_file = "data/processed/daily_sales_clean.csv"


df = pd.read_csv(input_file)


df["ds"] = pd.to_datetime(df["ds"])

df["y"] = pd.to_numeric(df["y"])


# replace zero sales
df["y"] = df["y"].replace(0, None)


# interpolate missing values
df["y"] = df["y"].interpolate(method="linear")


# fill remaining
df["y"] = df["y"].fillna(method="bfill")
df["y"] = df["y"].fillna(method="ffill")


df.to_csv(output_file,index=False)


print("Cleaning completed")
print(df.describe())