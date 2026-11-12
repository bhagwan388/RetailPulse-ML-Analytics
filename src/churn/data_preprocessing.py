import pandas as pd
from pathlib import Path

# ==========================================
# Paths
# ==========================================

DATA_PATH = Path("data/processed/online_retail_featured.csv")
OUTPUT_PATH = Path("data/processed/churn_cleaned.csv")

print("Loading dataset...")

# Load dataset
df = pd.read_csv(DATA_PATH, low_memory=False)

print(f"Original Shape : {df.shape}")

# ==========================================
# Check required columns
# ==========================================

required_columns = [
    "Customer ID",
    "Invoice",
    "InvoiceDate",
    "Quantity",
    "Price"
]

missing = [col for col in required_columns if col not in df.columns]

if len(missing) > 0:
    raise Exception(f"Missing columns: {missing}")

# ==========================================
# Remove missing Customer IDs
# ==========================================

df = df.dropna(subset=["Customer ID"])

# ==========================================
# Remove cancelled invoices
# ==========================================

df = df[
    ~df["Invoice"].astype(str).str.startswith("C")
]

# ==========================================
# Remove invalid Quantity
# ==========================================

df = df[df["Quantity"] > 0]

# ==========================================
# Remove invalid Price
# ==========================================

df = df[df["Price"] > 0]

# ==========================================
# Convert date
# ==========================================

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# ==========================================
# Create TotalPrice
# ==========================================

df["TotalPrice"] = (
    df["Quantity"] *
    df["Price"]
)

# ==========================================
# Keep required columns
# ==========================================

df = df[
    [
        "Customer ID",
        "Invoice",
        "InvoiceDate",
        "Quantity",
        "Price",
        "TotalPrice"
    ]
]

print(f"Cleaned Shape : {df.shape}")

# ==========================================
# Dataset Information
# ==========================================

print("\nDataset Summary")
print("-------------------------")
print(df.info())

print("\nMissing Values")
print("-------------------------")
print(df.isnull().sum())

print("\nFirst Five Rows")
print("-------------------------")
print(df.head())

# ==========================================
# Save dataset
# ==========================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nClean dataset saved to:")
print(OUTPUT_PATH)

print("\nData preprocessing completed successfully.")