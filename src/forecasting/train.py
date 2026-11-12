"""
RetailPulse - Demand Forecasting
Model Training Module

This module trains a Prophet time-series forecasting model
using the preprocessed daily sales dataset.

Input:
    data/processed/daily_sales.csv

Output:
    models/forecasting/prophet_model.pkl
"""

from pathlib import Path
import joblib
import pandas as pd
from prophet import Prophet

from utils import (
    logger,
    load_csv,
    get_processed_data_path,
    get_model_path,
    print_banner,
)


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "daily_sales.csv"
MODEL_FILE = "prophet_model.pkl"

YEARLY_SEASONALITY = True
WEEKLY_SEASONALITY = True
DAILY_SEASONALITY = False

CHANGEPOINT_PRIOR_SCALE = 0.05
SEASONALITY_MODE = "multiplicative"


# ============================================================
# Training Function
# ============================================================

def train_prophet_model():
    """
    Train the Prophet demand forecasting model.

    Returns
    -------
    Prophet
        Trained Prophet model.
    """

    print_banner("RetailPulse - Demand Forecasting Model Training")

    # --------------------------------------------------------
    # Define paths
    # --------------------------------------------------------

    input_path = get_processed_data_path(INPUT_FILE)
    model_path = get_model_path(MODEL_FILE)

    logger.info(f"Training dataset: {input_path}")
    logger.info(f"Model output: {model_path}")

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not input_path.exists():
        raise FileNotFoundError(
            f"\nTraining dataset not found!\n"
            f"Expected location:\n{input_path}\n\n"
            f"Run data_preprocessing.py first."
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    logger.info("Loading daily sales dataset...")

    df = load_csv(input_path)

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_columns = ["ds", "y"]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required Prophet columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    logger.info("Preparing training data...")

    df["ds"] = pd.to_datetime(
        df["ds"],
        errors="coerce"
    )

    df["y"] = pd.to_numeric(
        df["y"],
        errors="coerce"
    )

    # Remove invalid rows
    df = df.dropna(
        subset=["ds", "y"]
    )

    # Sort chronologically
    df = df.sort_values("ds")

    # Remove duplicate dates
    df = df.drop_duplicates(
        subset=["ds"]
    )

    # --------------------------------------------------------
    # Validate training data
    # --------------------------------------------------------

    if len(df) < 30:
        raise ValueError(
            "Not enough historical data for forecasting. "
            "At least 30 days are recommended."
        )

    logger.info(
        f"Training records: {len(df):,}"
    )

    logger.info(
        f"Training period: "
        f"{df['ds'].min().date()} → "
        f"{df['ds'].max().date()}"
    )

    # --------------------------------------------------------
    # Create Prophet model
    # --------------------------------------------------------

    logger.info("Creating Prophet model...")

    model = Prophet(
        yearly_seasonality=YEARLY_SEASONALITY,
        weekly_seasonality=WEEKLY_SEASONALITY,
        daily_seasonality=DAILY_SEASONALITY,
        changepoint_prior_scale=CHANGEPOINT_PRIOR_SCALE,
        seasonality_mode=SEASONALITY_MODE,
        interval_width=0.95,
    )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    logger.info("Training Prophet model...")

    model.fit(df)

    logger.info("Prophet model training completed.")

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        model_path
    )

    logger.info(
        f"Model saved successfully: {model_path}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED")
    print("=" * 60)

    print(f"\nTraining records : {len(df):,}")
    print(
        f"Start date      : "
        f"{df['ds'].min().date()}"
    )
    print(
        f"End date        : "
        f"{df['ds'].max().date()}"
    )

    print("\nModel Configuration")
    print("-" * 40)
    print(f"Model                   : Prophet")
    print(f"Yearly seasonality      : {YEARLY_SEASONALITY}")
    print(f"Weekly seasonality      : {WEEKLY_SEASONALITY}")
    print(f"Daily seasonality       : {DAILY_SEASONALITY}")
    print(
        f"Changepoint prior       : "
        f"{CHANGEPOINT_PRIOR_SCALE}"
    )
    print(
        f"Seasonality mode        : "
        f"{SEASONALITY_MODE}"
    )

    print("\nModel saved to:")
    print(model_path)

    return model


# ============================================================
# Script Entry Point
# ============================================================

if __name__ == "__main__":

    try:

        train_prophet_model()

    except Exception as error:

        logger.exception(
            f"Model training failed: {error}"
        )

        raise