from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

ROOT = Path("demo_data")
PRED = ROOT / "predictions"
REPORT = ROOT / "reports"

PRED.mkdir(parents=True, exist_ok=True)
REPORT.mkdir(parents=True, exist_ok=True)

N = 1000

customer_ids = np.arange(900001, 900001 + N)

# Artificial customer probabilities and outcomes.
probabilities = rng.beta(2.5, 2.0, N)
outcomes = rng.binomial(1, probabilities)
predicted = (probabilities >= 0.5).astype(int)

customers = pd.DataFrame({
    "CustomerID": customer_ids,
    "SnapshotDate": "2011-09-10",
    "Churn": outcomes,
    "ChurnProbability": probabilities,
    "PredictedChurn": predicted,
})

customers.to_csv(
    PRED / "churn_predictions_demo.csv",
    index=False
)

features = [
    "Recency",
    "Frequency",
    "Monetary",
    "TotalQuantity",
    "AvgOrderValue",
    "CustomerLifetime",
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
    "LogMonetary",
    "LogFrequency",
    "LogQuantity",
    "LogCustomerLifetime",
    "LogRecency",
    "MonetaryFrequency",
    "MonetaryRecency",
    "FrequencyQuantity",
    "DailyRevenue",
    "DailyOrders",
    "RevenueEfficiency",
    "QuantityEfficiency",
    "ActivityScore",
    "LoyaltyScore",
    "RevenueDensity",
]

# Simulated feature contributions for UI demonstration.
shap_values = rng.normal(
    loc=0,
    scale=0.12,
    size=(N, len(features))
)

shap_df = pd.DataFrame(
    shap_values,
    columns=features
)
shap_df.insert(0, "CustomerID", customer_ids)

shap_df.to_csv(
    REPORT / "customer_shap_values_demo.csv",
    index=False
)

importance = (
    shap_df[features]
    .abs()
    .mean()
    .sort_values(ascending=False)
)

importance_df = pd.DataFrame({
    "Feature": importance.index,
    "MeanAbsSHAP": importance.values,
    "ImportancePercent": (
        importance.values / importance.sum() * 100
    )
})

importance_df.to_csv(
    REPORT / "shap_feature_importance_demo.csv",
    index=False
)

print("===== SYNTHETIC DEMO GENERATED =====")
print("Customers:", len(customers))
print("SHAP rows:", len(shap_df))
print("Features:", len(features))
print("Synthetic churn rate:", round(outcomes.mean() * 100, 2), "%")
print("Files saved under:", ROOT)
