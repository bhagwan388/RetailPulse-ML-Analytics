"""
RetailPulse - Demand Forecasting Utilities

This module contains helper functions used across the
Demand Forecasting pipeline.

Author: Team RetailPulse
"""

import logging
from pathlib import Path
import pandas as pd


# -----------------------------
# Logging Configuration
# -----------------------------
def setup_logger():
    """
    Configure logging for the forecasting module.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger(__name__)


logger = setup_logger()


# -----------------------------
# Project Paths
# -----------------------------
def get_project_root():
    """
    Returns the project root directory.
    """
    return Path(__file__).resolve().parents[2]


def get_processed_data_path(filename):
    """
    Returns the path to a processed dataset.

    Example:
        get_processed_data_path("daily_sales.csv")
    """
    return get_project_root() / "data" / "processed" / filename


def get_model_path(filename):
    """
    Returns the model save path.

    Example:
        get_model_path("prophet_model.pkl")
    """
    return get_project_root() / "models" / "forecasting" / filename


def get_prediction_path(filename):
    """
    Returns the prediction output path.
    """
    return get_project_root() / "outputs" / "predictions" / filename


def get_report_path(filename):
    """
    Returns the reports folder path.
    """
    return get_project_root() / "outputs" / "reports" / filename


def get_figure_path(filename):
    """
    Returns the figures folder path.
    """
    return get_project_root() / "outputs" / "figures" / filename


# -----------------------------
# CSV Utilities
# -----------------------------
def load_csv(file_path):
    """
    Load a CSV file safely.

    Parameters
    ----------
    file_path : Path

    Returns
    -------
    pandas.DataFrame
    """

    try:
        df = pd.read_csv(file_path, low_memory=False)
        logger.info(f"Loaded dataset: {file_path}")
        logger.info(f"Shape: {df.shape}")
        return df

    except FileNotFoundError:
        logger.error(f"Dataset not found: {file_path}")
        raise

    except Exception as e:
        logger.error(e)
        raise


def save_csv(df, file_path):
    """
    Save DataFrame as CSV.
    """

    df.to_csv(file_path, index=False)

    logger.info(f"Saved file: {file_path}")


# -----------------------------
# Forecast Helpers
# -----------------------------
def classify_demand(revenue):
    """
    Convert forecast revenue into demand category.
    """

    if revenue >= 50000:
        return "High"

    elif revenue >= 25000:
        return "Medium"

    return "Low"


def clip_negative_forecast(df):
    """
    Replace negative forecast values with zero.
    """

    columns = [
        "Forecast_Revenue",
        "Lower_Bound",
        "Upper_Bound",
    ]

    for col in columns:

        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    return df


# -----------------------------
# Display Helpers
# -----------------------------
def print_banner(title):
    """
    Print a formatted title.
    """

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60 + "\n")