import pandas as pd
from pathlib import Path

# ==========================================
# Paths
# ==========================================

INPUT_PATH = Path("data/processed/churn_cleaned.csv")
OUTPUT_PATH = Path("data/processed/churn_features.csv")

print("Loading cleaned dataset...")

df = pd.read_csv(INPUT_PATH)

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print("Dataset Shape:", df.shape)

# ==========================================
# Time-based Split
# ==========================================

cutoff_date = pd.Timestamp("2011-09-10")

print("Cutoff Date:", cutoff_date.date())

train_df = df[df["InvoiceDate"] <= cutoff_date].copy()

future_df = df[df["InvoiceDate"] > cutoff_date].copy()

print("Training Transactions :", len(train_df))
print("Future Transactions   :", len(future_df))

# ==========================================
# Reference Date
# ==========================================

reference_date = cutoff_date + pd.Timedelta(days=1)

# ==========================================
# RFM Features
# ==========================================

rfm = train_df.groupby("Customer ID").agg(
    Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
    Frequency=("Invoice", "nunique"),
    Monetary=("TotalPrice", "sum"),
    TotalQuantity=("Quantity", "sum")
).reset_index()

rfm.rename(columns={"Customer ID": "CustomerID"}, inplace=True)

# ==========================================
# Average Order Value
# ==========================================

rfm["AvgOrderValue"] = (
    rfm["Monetary"] /
    rfm["Frequency"]
)

# ==========================================
# Customer Lifetime
# ==========================================

lifetime = train_df.groupby("Customer ID").agg(
    FirstPurchase=("InvoiceDate", "min"),
    LastPurchase=("InvoiceDate", "max")
).reset_index()

lifetime["CustomerLifetime"] = (
    lifetime["LastPurchase"] -
    lifetime["FirstPurchase"]
).dt.days

lifetime = lifetime[
    ["Customer ID", "CustomerLifetime"]
]

lifetime.rename(
    columns={
        "Customer ID": "CustomerID"
    },
    inplace=True
)

rfm = rfm.merge(
    lifetime,
    on="CustomerID",
    how="left"
)

# ==========================================
# Future Churn Label
# ==========================================

future_customers = set(
    future_df["Customer ID"].unique()
)

rfm["Churn"] = (
    ~rfm["CustomerID"].isin(future_customers)
).astype(int)

# ==========================================
# Save
# ==========================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

rfm.to_csv(
    OUTPUT_PATH,
    index=False
)

# ==========================================
# Summary
# ==========================================

print("\nFeature Engineering Completed")
print("-----------------------------")

print(rfm.head())

print("\nDataset Shape:", rfm.shape)

print("\nChurn Distribution")

print(rfm["Churn"].value_counts())

print("\nSaved to:")

print(OUTPUT_PATH)