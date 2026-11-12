"""
RetailPulse - Demand Forecasting
Model Evaluation Module

This module evaluates the trained Prophet model using
historical retail sales data.

Metrics:
    - MAE
    - RMSE
    - MAPE

Outputs:
    outputs/reports/forecast_metrics.txt
    outputs/figures/forecast_plot.png
    outputs/figures/trend_components.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet

from utils import (
    logger,
    get_model_path,
    get_processed_data_path,
    get_report_path,
    get_figure_path,
    load_csv,
    print_banner,
)


# ============================================================
# Configuration
# ============================================================

MODEL_FILE = "prophet_model.pkl"
INPUT_FILE = "daily_sales.csv"

METRICS_FILE = "forecast_metrics.txt"
FORECAST_PLOT_FILE = "forecast_plot.png"
COMPONENTS_PLOT_FILE = "trend_components.png"

TEST_DAYS = 30


# ============================================================
# Load Model
# ============================================================

def load_trained_model():
    """
    Load the trained Prophet model.
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

    logger.info("Model loaded successfully.")

    return model


# ============================================================
# Load Historical Data
# ============================================================

def load_historical_data():
    """
    Load the daily sales dataset.
    """

    input_path = get_processed_data_path(INPUT_FILE)

    if not input_path.exists():
        raise FileNotFoundError(
            f"\nDaily sales dataset not found!\n"
            f"Expected location:\n{input_path}\n\n"
            f"Run data_preprocessing.py first."
        )

    logger.info("Loading historical sales data...")

    df = load_csv(input_path)

    required_columns = ["ds", "y"]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["ds"] = pd.to_datetime(
        df["ds"],
        errors="coerce"
    )

    df["y"] = pd.to_numeric(
        df["y"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["ds", "y"]
    )

    df = df.sort_values("ds")

    df = df.drop_duplicates(
        subset=["ds"]
    )

    return df


# ============================================================
# Calculate MAPE
# ============================================================

def calculate_mape(actual, predicted):
    """
    Calculate Mean Absolute Percentage Error.

    Zero actual values are excluded to avoid division by zero.
    """

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    non_zero_mask = actual != 0

    if non_zero_mask.sum() == 0:
        return np.nan

    actual_filtered = actual[non_zero_mask]
    predicted_filtered = predicted[non_zero_mask]

    mape = np.mean(
        np.abs(
            (actual_filtered - predicted_filtered)
            / actual_filtered
        )
    ) * 100

    return mape


# ============================================================
# Evaluate Model
# ============================================================

def evaluate_model(model, df):
    """
    Evaluate Prophet model on the last TEST_DAYS
    observations.

    Returns:
        Dictionary containing evaluation metrics.
    """

    if len(df) <= TEST_DAYS:
        raise ValueError(
            f"Dataset must contain more than "
            f"{TEST_DAYS} observations."
        )

    logger.info(
        f"Using last {TEST_DAYS} days as test data."
    )

    # --------------------------------------------------------
    # Split data
    # --------------------------------------------------------

    train_df = df.iloc[:-TEST_DAYS].copy()

    test_df = df.iloc[-TEST_DAYS:].copy()

    logger.info(
        f"Training observations: {len(train_df)}"
    )

    logger.info(
        f"Testing observations: {len(test_df)}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # The existing model was trained on the complete dataset.
    #
    # For a proper evaluation, train a separate model using
    # only the training portion.
    # --------------------------------------------------------

    logger.info(
        "Training evaluation model on training portion..."
    )

    evaluation_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_mode="multiplicative",
        interval_width=0.95,
    )

    evaluation_model.fit(
        train_df
    )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    future = test_df[["ds"]].copy()

    forecast = evaluation_model.predict(
        future
    )

    # --------------------------------------------------------
    # Extract values
    # --------------------------------------------------------

    actual = test_df["y"].values

    predicted = forecast["yhat"].values

    # Prevent negative predicted revenue
    predicted = np.maximum(
        predicted,
        0
    )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    mape = calculate_mape(
        actual,
        predicted
    )

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
    }

    logger.info(
        f"MAE: {mae:.2f}"
    )

    logger.info(
        f"RMSE: {rmse:.2f}"
    )

    if np.isnan(mape):
        logger.info(
            "MAPE: Not available"
        )
    else:
        logger.info(
            f"MAPE: {mape:.2f}%"
        )

    return (
        metrics,
        test_df,
        forecast,
        evaluation_model
    )


# ============================================================
# Save Metrics
# ============================================================

def save_metrics(metrics, test_df):
    """
    Save evaluation metrics to a text report.
    """

    output_path = get_report_path(
        METRICS_FILE
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    mape = metrics["MAPE"]

    if np.isnan(mape):
        mape_text = "Not available"
    else:
        mape_text = f"{mape:.2f}%"

    report = f"""
RetailPulse - Demand Forecasting
Model Evaluation Report
========================================

Model:
Prophet

Evaluation Period:
{test_df['ds'].min().date()} to {test_df['ds'].max().date()}

Test Days:
{len(test_df)}

Metrics
----------------------------------------

MAE:
{metrics['MAE']:.2f}

RMSE:
{metrics['RMSE']:.2f}

MAPE:
{mape_text}

Interpretation
----------------------------------------

MAE:
Average absolute forecasting error.

RMSE:
Penalizes larger forecasting errors more heavily.

MAPE:
Average percentage forecasting error,
excluding days with zero actual revenue.

========================================
"""

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    logger.info(
        f"Metrics report saved to: {output_path}"
    )

    return output_path


# ============================================================
# Create Forecast Plot
# ============================================================

def create_forecast_plot(
    test_df,
    forecast
):
    """
    Create actual vs predicted revenue plot.
    """

    output_path = get_figure_path(
        FORECAST_PLOT_FILE
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        test_df["ds"],
        test_df["y"],
        label="Actual Revenue"
    )

    plt.plot(
        test_df["ds"],
        forecast["yhat"],
        label="Forecast Revenue"
    )

    plt.fill_between(
        test_df["ds"],
        forecast["yhat_lower"],
        forecast["yhat_upper"],
        alpha=0.2,
        label="Confidence Interval"
    )

    plt.title(
        "RetailPulse - Demand Forecast Evaluation"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Revenue"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    logger.info(
        f"Forecast plot saved to: {output_path}"
    )

    return output_path


# ============================================================
# Create Prophet Components Plot
# ============================================================

def create_components_plot(
    model,
    historical_df
):
    """
    Generate Prophet trend and seasonality
    component plots.
    """

    output_path = get_figure_path(
        COMPONENTS_PLOT_FILE
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    logger.info(
        "Generating Prophet component plots..."
    )

    future = model.make_future_dataframe(
        periods=0,
        freq="D"
    )

    forecast = model.predict(
        future
    )

    fig = model.plot_components(
        forecast
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    logger.info(
        f"Component plot saved to: {output_path}"
    )

    return output_path


# ============================================================
# Main Evaluation Pipeline
# ============================================================

def run_evaluation():

    print_banner(
        "RetailPulse - Demand Forecasting Evaluation"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_historical_data()

    logger.info(
        f"Historical records: {len(df):,}"
    )

    # --------------------------------------------------------
    # Load production model
    # --------------------------------------------------------

    production_model = load_trained_model()

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    (
        metrics,
        test_df,
        forecast,
        evaluation_model
    ) = evaluate_model(
        production_model,
        df
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_path = save_metrics(
        metrics,
        test_df
    )

    # --------------------------------------------------------
    # Create forecast plot
    # --------------------------------------------------------

    forecast_plot_path = create_forecast_plot(
        test_df,
        forecast
    )

    # --------------------------------------------------------
    # Create components plot
    # --------------------------------------------------------

    components_plot_path = create_components_plot(
        production_model,
        df
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL EVALUATION COMPLETED")
    print("=" * 60)

    print("\nEvaluation Period:")
    print(
        f"{test_df['ds'].min().date()} "
        f"→ "
        f"{test_df['ds'].max().date()}"
    )

    print("\nEvaluation Metrics")
    print("-" * 40)

    print(
        f"MAE  : "
        f"{metrics['MAE']:,.2f}"
    )

    print(
        f"RMSE : "
        f"{metrics['RMSE']:,.2f}"
    )

    if np.isnan(metrics["MAPE"]):
        print(
            "MAPE : Not available"
        )
    else:
        print(
            f"MAPE : "
            f"{metrics['MAPE']:.2f}%"
        )

    print("\nGenerated Files")
    print("-" * 40)

    print(
        f"Metrics report : "
        f"{metrics_path}"
    )

    print(
        f"Forecast plot  : "
        f"{forecast_plot_path}"
    )

    print(
        f"Components plot: "
        f"{components_plot_path}"
    )

    return metrics


# ============================================================
# Script Entry Point
# ============================================================

if __name__ == "__main__":

    try:

        run_evaluation()

    except Exception as error:

        logger.exception(
            f"Model evaluation failed: {error}"
        )

        raise