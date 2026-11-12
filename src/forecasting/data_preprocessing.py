"""
RetailPulse - Demand Forecasting
Data Preprocessing Module

This module converts transaction-level retail data into
daily time-series data suitable for demand forecasting.

Input:
    data/processed/online_retail_featured.csv

Output:
    data/processed/daily_sales.csv
"""

from pathlib import Path
import pandas as pd

from utils import (
    logger,
    load_csv,
    save_csv,
    print_banner,
    get_processed_data_path,
)


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "online_retail_featured.csv"
OUTPUT_FILE = "daily_sales.csv"


# ============================================================
# Main Preprocessing Function
# ============================================================

def prepare_daily_sales():
    """
    Prepare daily sales data for the forecasting model.

    Steps:
        1. Load featured retail dataset.
        2. Validate required columns.
        3. Convert InvoiceDate to datetime.
        4. Remove invalid dates.
        5. Handle missing revenue values.
        6. Aggregate revenue by date.
        7. Add missing dates with zero revenue.
        8. Sort chronologically.
        9. Save the processed time-series dataset.

    Returns
    -------
    pandas.DataFrame
        Clean daily sales dataset.
    """

    print_banner("RetailPulse - Demand Forecasting Preprocessing")

    # --------------------------------------------------------
    # Define input/output paths
    # --------------------------------------------------------

    input_path = get_processed_data_path(INPUT_FILE)
    output_path = get_processed_data_path(OUTPUT_FILE)

    logger.info("Starting demand forecasting preprocessing...")
    logger.info(f"Input file: {input_path}")

    # --------------------------------------------------------
    # Check input file
    # --------------------------------------------------------

    if not input_path.exists():
        raise FileNotFoundError(
            f"\nInput dataset not found!\n"
            f"Expected location:\n{input_path}\n"
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_csv(input_path)

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_columns = [
        "InvoiceDate",
        "Revenue",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    logger.info("Required columns validated successfully.")

    # --------------------------------------------------------
    # Convert InvoiceDate to datetime
    # --------------------------------------------------------

    logger.info("Converting InvoiceDate to datetime...")

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )

    # Remove rows where date conversion failed
    invalid_dates = df["InvoiceDate"].isna().sum()

    if invalid_dates > 0:
        logger.warning(
            f"Removing {invalid_dates} rows with invalid dates."
        )

        df = df.dropna(subset=["InvoiceDate"])

    # --------------------------------------------------------
    # Handle Revenue
    # --------------------------------------------------------

    logger.info("Processing Revenue column...")

    df["Revenue"] = pd.to_numeric(
        df["Revenue"],
        errors="coerce"
    )

    invalid_revenue = df["Revenue"].isna().sum()

    if invalid_revenue > 0:
        logger.warning(
            f"Replacing {invalid_revenue} invalid revenue values with 0."
        )

        df["Revenue"] = df["Revenue"].fillna(0)

    # --------------------------------------------------------
    # Create Date column
    # --------------------------------------------------------

    df["Date"] = df["InvoiceDate"].dt.normalize()

    # --------------------------------------------------------
    # Aggregate revenue by day
    # --------------------------------------------------------

    logger.info("Aggregating revenue by day...")

    daily_sales = (
        df.groupby("Date", as_index=False)["Revenue"]
        .sum()
    )

    # --------------------------------------------------------
    # Sort by date
    # --------------------------------------------------------

    daily_sales = daily_sales.sort_values("Date")

    # --------------------------------------------------------
    # Create continuous date range
    # --------------------------------------------------------

    logger.info("Checking for missing dates...")

    full_date_range = pd.date_range(
        start=daily_sales["Date"].min(),
        end=daily_sales["Date"].max(),
        freq="D"
    )

    daily_sales = (
        daily_sales
        .set_index("Date")
        .reindex(full_date_range)
        .rename_axis("Date")
        .reset_index()
    )

    # Missing dates mean no recorded sales.
    daily_sales["Revenue"] = daily_sales["Revenue"].fillna(0)

    # --------------------------------------------------------
    # Rename columns for Prophet
    # --------------------------------------------------------

    daily_sales = daily_sales.rename(
        columns={
            "Date": "ds",
            "Revenue": "y"
        }
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    daily_sales["ds"] = pd.to_datetime(
        daily_sales["ds"]
    )

    daily_sales["y"] = pd.to_numeric(
        daily_sales["y"],
        errors="coerce"
    ).fillna(0)

    daily_sales = daily_sales.sort_values("ds")

    # Remove duplicate dates if any
    daily_sales = daily_sales.drop_duplicates(
        subset=["ds"]
    )

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    save_csv(
        daily_sales,
        output_path
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\nPreprocessing completed successfully!")

    print("\nDataset Summary")
    print("-" * 40)

    print(f"Total days       : {len(daily_sales):,}")
    print(f"Start date       : {daily_sales['ds'].min().date()}")
    print(f"End date         : {daily_sales['ds'].max().date()}")
    print(
        f"Total revenue    : "
        f"{daily_sales['y'].sum():,.2f}"
    )
    print(
        f"Average daily    : "
        f"{daily_sales['y'].mean():,.2f}"
    )

    print("\nFirst 5 records:")
    print(daily_sales.head())

    print("\nLast 5 records:")
    print(daily_sales.tail())

    print(f"\nOutput saved to:")
    print(output_path)

    return daily_sales


# ============================================================
# Script Entry Point
# ============================================================

if __name__ == "__main__":

    try:

        prepare_daily_sales()

    except Exception as error:

        logger.exception(
            f"Preprocessing failed: {error}"
        )

        raise