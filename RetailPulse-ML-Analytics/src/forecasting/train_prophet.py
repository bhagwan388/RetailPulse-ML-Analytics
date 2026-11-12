import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_PATH = "data/processed/daily_sales.csv"

OUTPUT_PATH = "outputs/predictions/prophet_forecast.csv"
FIG_PATH = "outputs/figures/prophet_forecast.png"


print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

df["ds"] = pd.to_datetime(df["ds"])
df = df.sort_values("ds")


print("Dataset size:", len(df))
print("Last training date:", df["ds"].max())


prophet_df = df.rename(
    columns={
        "y": "y"
    }
)


model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=5
)


model.fit(prophet_df)


future = model.make_future_dataframe(
    periods=30,
    freq="D"
)


forecast = model.predict(future)


future_forecast = forecast[
    forecast["ds"] > df["ds"].max()
][
    [
        "ds",
        "yhat",
        "yhat_lower",
        "yhat_upper"
    ]
]


future_forecast.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n30 Day Prophet Forecast Saved")


plt.figure(figsize=(12,5))

plt.plot(
    future_forecast["ds"],
    future_forecast["yhat"]
)

plt.title(
    "Prophet 30 Day Demand Forecast"
)

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(FIG_PATH)

plt.close()


print("Plot saved:", FIG_PATH)


print("Prophet training completed successfully")