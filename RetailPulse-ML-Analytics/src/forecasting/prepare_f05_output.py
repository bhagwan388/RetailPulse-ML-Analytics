"""
RetailPulse - F03 Demand Forecasting
F03 -> F05 Output Preparation

Creates a clean demand forecast file that can be
used by the Inventory Optimization module.
"""

from pathlib import Path
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "forecasted_demand.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "f05_inventory_demand_input.csv"
)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("RetailPulse - F03 → F05 Output Preparation")
    print("=" * 60)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nForecast file not found:\n{INPUT_FILE}\n\n"
            "Run predict.py first."
        )

    print("\nLoading forecast file...")

    df = pd.read_csv(INPUT_FILE)

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    required_columns = [
        "Date",
        "Forecast_Revenue",
        "Lower_Bound",
        "Upper_Bound",
        "Forecast_Status",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # Clean data
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    numeric_columns = [
        "Forecast_Revenue",
        "Lower_Bound",
        "Upper_Bound",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "Date",
            "Forecast_Revenue"
        ]
    )

    # --------------------------------------------------------
    # Prevent negative demand values
    # --------------------------------------------------------

    df["Forecast_Revenue"] = (
        df["Forecast_Revenue"]
        .clip(lower=0)
    )

    df["Lower_Bound"] = (
        df["Lower_Bound"]
        .clip(lower=0)
    )

    df["Upper_Bound"] = (
        df["Upper_Bound"]
        .clip(lower=0)
    )

    # --------------------------------------------------------
    # Add useful inventory fields
    # --------------------------------------------------------

    df["Forecast_Month"] = (
        df["Date"]
        .dt.to_period("M")
        .astype(str)
    )

    df["Forecast_Day"] = (
        df["Date"]
        .dt.day_name()
    )

    df["Weekend"] = (
        df["Date"]
        .dt.dayofweek >= 5
    )

    # --------------------------------------------------------
    # Rename revenue forecast for F05
    # --------------------------------------------------------

    df = df.rename(
        columns={
            "Forecast_Revenue": "Forecast_Demand",
            "Lower_Bound": "Demand_Lower_Bound",
            "Upper_Bound": "Demand_Upper_Bound",
        }
    )

    # --------------------------------------------------------
    # Select final columns
    # --------------------------------------------------------

    final_columns = [
        "Date",
        "Forecast_Demand",
        "Demand_Lower_Bound",
        "Demand_Upper_Bound",
        "Forecast_Status",
        "Forecast_Month",
        "Forecast_Day",
        "Weekend",
    ]

    df = df[final_columns]

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = df.sort_values(
        "Date"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\nF03 → F05 file created successfully!")

    print("\nOutput:")
    print(OUTPUT_FILE)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nPreview:")
    print(df.head())

    print("\nForecast Summary")
    print("-" * 40)

    print(
        f"Total forecast demand : "
        f"{df['Forecast_Demand'].sum():,.2f}"
    )

    print(
        f"Average daily demand  : "
        f"{df['Forecast_Demand'].mean():,.2f}"
    )

    print(
        f"Forecast days         : "
        f"{len(df)}"
    )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()