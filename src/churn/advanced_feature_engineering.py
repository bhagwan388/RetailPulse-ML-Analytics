# ============================================================
# RetailPulse
# F4 Customer Churn Prediction
# Advanced Feature Engineering
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# Create Output Folders
# ============================================================

os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

print("=" * 60)
print("ADVANCED FEATURE ENGINEERING")
print("=" * 60)

# ============================================================
# Load Dataset
# ============================================================

print("\nLoading feature engineered dataset...")

df = pd.read_csv("data/processed/churn_features.csv")

print(f"Dataset Shape : {df.shape}")

# ============================================================
# Required Columns
# ============================================================

required_columns = [

    "CustomerID",
    "Recency",
    "Frequency",
    "Monetary",
    "TotalQuantity",
    "AvgOrderValue",
    "CustomerLifetime",
    "Churn",

]

missing_columns = [

    column
    for column in required_columns
    if column not in df.columns

]

if len(missing_columns) > 0:

    raise ValueError(
        f"Missing Columns : {missing_columns}"
    )

print("\nAll required columns found.")

# ============================================================
# Replace Invalid Values
# ============================================================

# Avoid divide-by-zero problems

df["Frequency"] = df["Frequency"].replace(0, 1)

df["CustomerLifetime"] = df["CustomerLifetime"].replace(0, 1)

df["TotalQuantity"] = df["TotalQuantity"].replace(0, 1)

print("Zero values handled.")

# ============================================================
# Basic Information
# ============================================================

print("\nDataset Summary")

print(df.describe())

# ============================================================
# ADVANCED FEATURE 1
# Purchase Frequency
# Orders per day
# ============================================================

df["PurchaseFrequency"] = (

    df["Frequency"] /

    df["CustomerLifetime"]

)

# ============================================================
# ADVANCED FEATURE 2
# Average Quantity Per Order
# ============================================================

df["AvgQuantityPerOrder"] = (

    df["TotalQuantity"]

    /

    df["Frequency"]

)

# ============================================================
# ADVANCED FEATURE 3
# Days Between Orders
# ============================================================

df["DaysPerOrder"] = (

    df["CustomerLifetime"]

    /

    df["Frequency"]

)

# ============================================================
# ADVANCED FEATURE 4
# Monetary Per Day
# ============================================================

df["MonetaryPerDay"] = (

    df["Monetary"]

    /

    df["CustomerLifetime"]

)

# ============================================================
# ADVANCED FEATURE 5
# Quantity Per Day
# ============================================================

df["QuantityPerDay"] = (

    df["TotalQuantity"]

    /

    df["CustomerLifetime"]

)

# ============================================================
# ADVANCED FEATURE 6
# Revenue Per Item
# ============================================================

df["RevenuePerItem"] = (

    df["Monetary"]

    /

    df["TotalQuantity"]

)

# ============================================================
# ADVANCED FEATURE 7
# Revenue Per Order
# ============================================================

df["RevenuePerOrder"] = (

    df["Monetary"]

    /

    df["Frequency"]

)

# ============================================================
# ADVANCED FEATURE 8
# Quantity Per Customer Lifetime
# ============================================================

df["LifetimeQuantityRate"] = (

    df["TotalQuantity"]

    /

    df["CustomerLifetime"]

)

# ============================================================
# ADVANCED FEATURE 9
# High Value Customer Score
# ============================================================

df["CustomerValueScore"] = (

    df["Monetary"]

    *

    df["Frequency"]

)

# ============================================================
# ADVANCED FEATURE 10
# Engagement Score
# ============================================================

df["EngagementScore"] = (

    df["Frequency"]

    /

    (df["Recency"] + 1)

)

print("\nFirst 10 advanced features created successfully.")

print("\nCurrent Dataset Shape")

print(df.shape)

print("\nNew Columns Added")

new_columns = [

    "PurchaseFrequency",
    "AvgQuantityPerOrder",
    "DaysPerOrder",
    "MonetaryPerDay",
    "QuantityPerDay",
    "RevenuePerItem",
    "RevenuePerOrder",
    "LifetimeQuantityRate",
    "CustomerValueScore",
    "EngagementScore",

]

for column in new_columns:

    print(f"✔ {column}")

print("\nProceeding to logarithmic and interaction features...")

# ============================================================
# ADVANCED FEATURE 11
# Log Monetary
# ============================================================

df["LogMonetary"] = np.log1p(df["Monetary"])

# ============================================================
# ADVANCED FEATURE 12
# Log Frequency
# ============================================================

df["LogFrequency"] = np.log1p(df["Frequency"])

# ============================================================
# ADVANCED FEATURE 13
# Log Quantity
# ============================================================

df["LogQuantity"] = np.log1p(df["TotalQuantity"])

# ============================================================
# ADVANCED FEATURE 14
# Log Customer Lifetime
# ============================================================

df["LogCustomerLifetime"] = np.log1p(
    df["CustomerLifetime"]
)

# ============================================================
# ADVANCED FEATURE 15
# Log Recency
# ============================================================

df["LogRecency"] = np.log1p(
    df["Recency"]
)

# ============================================================
# ADVANCED FEATURE 16
# Monetary × Frequency
# ============================================================

df["MonetaryFrequency"] = (

    df["Monetary"]

    *

    df["Frequency"]

)

# ============================================================
# ADVANCED FEATURE 17
# Monetary × Recency
# ============================================================

df["MonetaryRecency"] = (

    df["Monetary"]

    *

    df["Recency"]

)

# ============================================================
# ADVANCED FEATURE 18
# Frequency × Quantity
# ============================================================

df["FrequencyQuantity"] = (

    df["Frequency"]

    *

    df["TotalQuantity"]

)

# ============================================================
# ADVANCED FEATURE 19
# Average Revenue Generated Daily
# ============================================================

df["DailyRevenue"] = (

    df["Monetary"]

    /

    (df["CustomerLifetime"] + 1)

)

# ============================================================
# ADVANCED FEATURE 20
# Average Orders Generated Daily
# ============================================================

df["DailyOrders"] = (

    df["Frequency"]

    /

    (df["CustomerLifetime"] + 1)

)

# ============================================================
# ADVANCED FEATURE 21
# Revenue Efficiency
# ============================================================

df["RevenueEfficiency"] = (

    df["AvgOrderValue"]

    /

    (df["DaysPerOrder"] + 1)

)

# ============================================================
# ADVANCED FEATURE 22
# Quantity Efficiency
# ============================================================

df["QuantityEfficiency"] = (

    df["AvgQuantityPerOrder"]

    /

    (df["DaysPerOrder"] + 1)

)

# ============================================================
# ADVANCED FEATURE 23
# Customer Activity Score
# ============================================================

df["ActivityScore"] = (

    df["PurchaseFrequency"]

    *

    df["CustomerLifetime"]

)

# ============================================================
# ADVANCED FEATURE 24
# Customer Loyalty Score
# ============================================================

df["LoyaltyScore"] = (

    df["Frequency"]

    *

    df["CustomerLifetime"]

)

# ============================================================
# ADVANCED FEATURE 25
# Revenue Density
# ============================================================

df["RevenueDensity"] = (

    df["Monetary"]

    /

    (df["TotalQuantity"] + 1)

)

# ============================================================
# Handle Infinite Values
# ============================================================

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True,
)

# ============================================================
# Fill Missing Values
# ============================================================

numeric_columns = df.select_dtypes(
    include=[np.number]
).columns

for column in numeric_columns:

    df[column] = df[column].fillna(
        df[column].median()
    )

# ============================================================
# Round Numeric Values
# ============================================================

df[numeric_columns] = df[numeric_columns].round(4)

# ============================================================
# Save Advanced Dataset
# ============================================================

output_path = (
    "data/processed/advanced_churn_features.csv"
)

df.to_csv(
    output_path,
    index=False,
)

print("\nAdvanced dataset saved successfully.")

print(output_path)

# ============================================================
# Feature Summary
# ============================================================

print("\nTotal Features")

print(df.shape[1] - 2)

print("\nDataset Shape")

print(df.shape)

print("\nAdvanced feature engineering completed.")


# ============================================================
# Feature Statistics
# ============================================================

print("\nGenerating feature statistics...")

statistics = df.describe().T

statistics.to_csv(
    "outputs/reports/feature_statistics.csv"
)

print("Feature statistics saved.")

# ============================================================
# Correlation Matrix
# ============================================================

print("\nGenerating correlation matrix...")

correlation = df.corr(numeric_only=True)

correlation.to_csv(
    "outputs/reports/correlation_matrix.csv"
)

print("Correlation matrix saved.")

# ============================================================
# Correlation Heatmap
# ============================================================

plt.figure(figsize=(16,12))

plt.imshow(
    correlation,
    cmap="coolwarm",
    aspect="auto"
)

plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90,
    fontsize=8
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns,
    fontsize=8
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "outputs/figures/correlation_heatmap.png",
    dpi=300
)

plt.close()

print("Correlation heatmap saved.")

# ============================================================
# Distribution Plots
# ============================================================

important_features = [

    "Recency",
    "Frequency",
    "Monetary",
    "PurchaseFrequency",
    "CustomerValueScore",
    "EngagementScore",
    "RevenuePerOrder",
    "RevenueEfficiency",

]

for feature in important_features:

    plt.figure(figsize=(6,4))

    plt.hist(
        df[feature],
        bins=30
    )

    plt.title(feature)

    plt.xlabel(feature)

    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig(
        f"outputs/figures/{feature}_distribution.png",
        dpi=300
    )

    plt.close()

print("Distribution plots saved.")

# ============================================================
# Boxplots
# ============================================================

for feature in important_features:

    plt.figure(figsize=(6,4))

    plt.boxplot(df[feature])

    plt.title(feature)

    plt.tight_layout()

    plt.savefig(
        f"outputs/figures/{feature}_boxplot.png",
        dpi=300
    )

    plt.close()

print("Boxplots saved.")

# ============================================================
# Feature Summary Report
# ============================================================

report_path = "outputs/reports/advanced_features_report.txt"

with open(report_path, "w") as report:

    report.write("RetailPulse\n")
    report.write("Advanced Feature Engineering Report\n")
    report.write("=" * 60)
    report.write("\n\n")

    report.write(f"Dataset Shape : {df.shape}\n\n")

    report.write("Generated Features\n")
    report.write("------------------------------\n")

    for column in df.columns:

        report.write(column + "\n")

    report.write("\n")

    report.write("Summary Statistics\n")
    report.write("------------------------------\n")

    report.write(statistics.to_string())

print("Feature report saved.")

# ============================================================
# Final Summary
# ============================================================

print("\n" + "="*60)
print("ADVANCED FEATURE ENGINEERING COMPLETED")
print("="*60)

print("\nGenerated Files")

generated_files = [

    "data/processed/advanced_churn_features.csv",

    "outputs/reports/feature_statistics.csv",

    "outputs/reports/correlation_matrix.csv",

    "outputs/reports/advanced_features_report.txt",

    "outputs/figures/correlation_heatmap.png",

]

for file in generated_files:

    print(f"✔ {file}")

print("\nDataset Information")
print("------------------------------")

print(f"Rows     : {df.shape[0]}")
print(f"Columns  : {df.shape[1]}")
print(f"New Features Added : {df.shape[1] - 8}")

print("\nAdvanced feature engineering completed successfully.")