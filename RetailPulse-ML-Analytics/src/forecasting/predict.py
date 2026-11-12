"""
RetailPulse - Demand Forecasting
Prediction Module

This module loads the trained Prophet model and generates
future revenue forecasts.

Input:
    models/forecasting/prophet_model.pkl

Output:
    outputs/predictions/forecasted_demand.csv
"""

import joblib
import pandas as pd

from utils import (
    logger,
    get_model_path,
    get_prediction_path,
    get_processed_data_path,
    save_csv,
    clip_negative_forecast,
    classify_demand,
    print_banner,
)


# ============================================================
# Configuration
# ============================================================

MODEL_FILE = "prophet_model.pkl"
INPUT_FILE = "daily_sales.csv"
OUTPUT_FILE = "forecasted_demand.csv"

# Number of future days to forecast
FORECAST_DAYS = 30


# ============================================================
# Load Trained Model
# ============================================================

def load_trained_model():
    """
    Load the trained Prophet model from disk.

    Returns
    -------
    Prophet
        Trained Prophet model.
    """

    model_path = get_model_path(MODEL_FILE)

    if not model_path.exists():
        raise FileNotFoundError(
            f"\nTrained model not found!\n"
            f"Expected location:\n{model_path}\n\n"
            f"Run train.py first."
        )

    logger.info("Loading trained Prophet model...")

    model = joblib.load(model_path)

    logger.info("Trained model loaded successfully.")

    return model


# ============================================================
# Load Historical Data
# ============================================================

def load_historical_data():
    """
    Load the historical daily sales dataset.

    Returns
    -------
    pandas.DataFrame
    """

    input_path = get_processed_data_path(INPUT_FILE)

    if not input_path.exists():
        raise FileNotFoundError(
            f"\nHistorical dataset not found!\n"
            f"Expected location:\n{input_path}\n\n"
            f"Run data_preprocessing.py first."
        )

    logger.info("Loading historical sales data...")

    df = pd.read_csv(input_path)

    if "ds" not in df.columns:
        raise ValueError(
            "Historical dataset must contain 'ds' column."
        )

    df["ds"] = pd.to_datetime(
        df["ds"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["ds"]
    )

    df = df.sort_values("ds")

    return df


# ============================================================
# Generate Forecast
# ============================================================

def generate_forecast(
    model,
    forecast_days=FORECAST_DAYS
):
    """
    Generate future revenue forecast.

    Parameters
    ----------
    model : Prophet
        Trained Prophet model.

    forecast_days : int
        Number of future days.

    Returns
    -------
    pandas.DataFrame
        Forecast dataframe.
    """

    if forecast_days <= 0:
        raise ValueError(
            "forecast_days must be greater than zero."
        )

    logger.info(
        f"Generating forecast for "
        f"{forecast_days} days..."
    )

    # Create future dates
    future = model.make_future_dataframe(
        periods=forecast_days,
        freq="D"
    )

    # Generate prediction
    forecast = model.predict(future)

    # Keep only future dates
    future_forecast = forecast.tail(
        forecast_days
    ).copy()

    # Select required columns
    future_forecast = future_forecast[
        [
            "ds",
            "yhat",
            "yhat_lower",
            "yhat_upper"
        ]
    ]

    # Rename columns
    future_forecast = future_forecast.rename(
        columns={
            "ds": "Date",
            "yhat": "Forecast_Revenue",
            "yhat_lower": "Lower_Bound",
            "yhat_upper": "Upper_Bound"
        }
    )

    # Convert date
    future_forecast["Date"] = pd.to_datetime(
        future_forecast["Date"]
    )

    # Prevent negative revenue forecasts
    future_forecast = clip_negative_forecast(
        future_forecast.rename(
            columns={
                "Forecast_Revenue": "Forecast_Revenue",
                "Lower_Bound": "Lower_Bound",
                "Upper_Bound": "Upper_Bound"
            }
        )
    )

    # Add demand classification
    future_forecast["Forecast_Status"] = (
        future_forecast["Forecast_Revenue"]
        .apply(classify_demand)
    )

    # Round numerical values
    future_forecast["Forecast_Revenue"] = (
        future_forecast["Forecast_Revenue"]
        .round(2)
    )

    future_forecast["Lower_Bound"] = (
        future_forecast["Lower_Bound"]
        .round(2)
    )

    future_forecast["Upper_Bound"] = (
        future_forecast["Upper_Bound"]
        .round(2)
    )

    return future_forecast


# ============================================================
# Save Forecast
# ============================================================

def save_forecast(forecast_df):
    """
    Save forecast dataframe as CSV.
    """

    output_path = get_prediction_path(
        OUTPUT_FILE
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    save_csv(
        forecast_df,
        output_path
    )

    return output_path


# ============================================================
# Main Prediction Pipeline
# ============================================================

def run_prediction():

    print_banner(
        "RetailPulse - Demand Forecasting Prediction"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_trained_model()

    # --------------------------------------------------------
    # Load historical data
    # --------------------------------------------------------

    historical_data = load_historical_data()

    logger.info(
        f"Historical records: "
        f"{len(historical_data):,}"
    )

    logger.info(
        f"Last historical date: "
        f"{historical_data['ds'].max().date()}"
    )

    # --------------------------------------------------------
    # Generate forecast
    # --------------------------------------------------------

    forecast_df = generate_forecast(
        model,
        FORECAST_DAYS
    )

    # --------------------------------------------------------
    # Save forecast
    # --------------------------------------------------------

    output_path = save_forecast(
        forecast_df
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FORECAST GENERATED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nForecast period : {FORECAST_DAYS} days")

    print(
        f"From            : "
        f"{forecast_df['Date'].min().date()}"
    )

    print(
        f"To              : "
        f"{forecast_df['Date'].max().date()}"
    )

    print("\nForecast Preview")
    print("-" * 60)

    print(
        forecast_df.to_string(
            index=False
        )
    )

    print("\nForecast Summary")
    print("-" * 60)

    total_forecast = (
        forecast_df["Forecast_Revenue"]
        .sum()
    )

    average_forecast = (
        forecast_df["Forecast_Revenue"]
        .mean()
    )

    high_days = (
        forecast_df["Forecast_Status"]
        .eq("High")
        .sum()
    )

    medium_days = (
        forecast_df["Forecast_Status"]
        .eq("Medium")
        .sum()
    )

    low_days = (
        forecast_df["Forecast_Status"]
        .eq("Low")
        .sum()
    )

    print(
        f"Total forecast revenue : "
        f"{total_forecast:,.2f}"
    )

    print(
        f"Average daily revenue  : "
        f"{average_forecast:,.2f}"
    )

    print(
        f"High demand days       : "
        f"{high_days}"
    )

    print(
        f"Medium demand days     : "
        f"{medium_days}"
    )

    print(
        f"Low demand days        : "
        f"{low_days}"
    )

    print("\nSaved to:")
    print(output_path)

    return forecast_df


# ============================================================
# Script Entry Point
# ============================================================

if __name__ == "__main__":

    try:

        run_prediction()

    except Exception as error:

        logger.exception(
            f"Forecast generation failed: {error}"
        )

        raise