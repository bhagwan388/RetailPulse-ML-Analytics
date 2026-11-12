"""
RetailPulse - Demand Forecasting
Configuration Module

Centralized configuration for the forecasting pipeline.
"""

# ============================================================
# File Names
# ============================================================

RAW_FEATURED_DATA = "online_retail_featured.csv"

DAILY_SALES_FILE = "daily_sales.csv"

MODEL_FILE = "prophet_model.pkl"

FORECAST_FILE = "forecasted_demand.csv"


# ============================================================
# Forecast Configuration
# ============================================================

# Number of future days to forecast
FORECAST_DAYS = 30


# ============================================================
# Prophet Configuration
# ============================================================

YEARLY_SEASONALITY = True

WEEKLY_SEASONALITY = True

DAILY_SEASONALITY = False

CHANGEPOINT_PRIOR_SCALE = 0.05

SEASONALITY_MODE = "multiplicative"

INTERVAL_WIDTH = 0.95


# ============================================================
# Demand Classification
# ============================================================

HIGH_DEMAND_THRESHOLD = 50000

MEDIUM_DEMAND_THRESHOLD = 25000


# ============================================================
# Evaluation Configuration
# ============================================================

TEST_DAYS = 30


# ============================================================
# Output Files
# ============================================================

METRICS_FILE = "forecast_metrics.txt"

FORECAST_PLOT = "forecast_plot.png"

COMPONENTS_PLOT = "trend_components.png"