from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/processed/churn_features.csv")
MODEL_PATH = Path("models/churn/final_churn_model.pkl")
FEATURES_PATH = Path("models/churn/final_churn_features.pkl")
OUTPUT_PATH = Path("outputs/predictions/churn_risk_predictions.csv")

CUSTOMER_ID_COLUMN = "CustomerID"
TARGET_COLUMN = "Churn"
PREDICTION_THRESHOLD = 0.50

# Risk-level thresholds
HIGH_RISK_THRESHOLD = 0.70
MEDIUM_RISK_THRESHOLD = 0.40


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def validate_file_exists(path: Path, description: str) -> None:
    """Raise a clear error when a required file does not exist."""
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path.resolve()}"
        )


def normalize_feature_list(features: object) -> list[str]:
    """
    Convert the saved feature object into a clean list of column names.

    The feature file may contain a Python list, tuple, NumPy array,
    Pandas Index, or another iterable object.
    """
    if isinstance(features, pd.Index):
        feature_list = features.tolist()
    elif isinstance(features, np.ndarray):
        feature_list = features.tolist()
    elif isinstance(features, (list, tuple)):
        feature_list = list(features)
    elif isinstance(features, Iterable) and not isinstance(features, (str, bytes)):
        feature_list = list(features)
    else:
        raise TypeError(
            "The saved feature file does not contain a valid feature list. "
            f"Received type: {type(features).__name__}"
        )

    feature_list = [str(feature) for feature in feature_list]

    if not feature_list:
        raise ValueError("The saved feature list is empty.")

    return feature_list


def assign_risk_level(probability: float) -> str:
    """Convert churn probability into a dashboard-friendly risk category."""
    if probability >= HIGH_RISK_THRESHOLD:
        return "High"
    if probability >= MEDIUM_RISK_THRESHOLD:
        return "Medium"
    return "Low"


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:
    print_section("F04 CHURN PREDICTION EXPORT")

    # --------------------------------------------------------
    # Validate required files
    # --------------------------------------------------------
    validate_file_exists(DATA_PATH, "Churn feature dataset")
    validate_file_exists(MODEL_PATH, "Final churn model")
    validate_file_exists(FEATURES_PATH, "Final feature list")

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------
    print("Loading churn feature dataset...")
    df = pd.read_csv(DATA_PATH)

    print(f"Dataset path  : {DATA_PATH}")
    print(f"Dataset shape : {df.shape}")
    print(f"Columns       : {df.columns.tolist()}")

    if df.empty:
        raise ValueError("The churn dataset is empty.")

    if CUSTOMER_ID_COLUMN not in df.columns:
        raise KeyError(
            f"Required customer ID column '{CUSTOMER_ID_COLUMN}' "
            f"was not found in {DATA_PATH}."
        )

    customer_ids = df[CUSTOMER_ID_COLUMN].copy()

    # --------------------------------------------------------
    # Load trained model and saved feature list
    # --------------------------------------------------------
    print_section("LOADING MODEL")

    print(f"Loading model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    print(f"Loading feature list from: {FEATURES_PATH}")
    saved_features = joblib.load(FEATURES_PATH)
    selected_features = normalize_feature_list(saved_features)

    print(f"Number of selected features: {len(selected_features)}")
    print("Selected features:")
    for index, feature in enumerate(selected_features, start=1):
        print(f"{index:02d}. {feature}")

    # --------------------------------------------------------
    # Validate model features
    # --------------------------------------------------------
    missing_features = [
        feature
        for feature in selected_features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "The prediction dataset is missing features required by "
            f"the trained model:\n{missing_features}\n\n"
            "Use the same processed dataset that was used during training."
        )

    X = df[selected_features].copy()

    # Convert all model features to numeric
    for column in X.columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    missing_value_counts = X.isna().sum()
    columns_with_missing_values = missing_value_counts[
        missing_value_counts > 0
    ]

    if not columns_with_missing_values.empty:
        print("\nWarning: Missing values were found in model features:")
        print(columns_with_missing_values)

        # Median imputation keeps the export script robust.
        # Ideally, use the exact preprocessing pipeline from training.
        X = X.fillna(X.median(numeric_only=True))

    if X.isna().any().any():
        problematic_columns = X.columns[X.isna().any()].tolist()
        raise ValueError(
            "Missing values remain after median imputation in columns: "
            f"{problematic_columns}"
        )

    # Match model feature order when supported by scikit-learn/XGBoost
    if hasattr(model, "feature_names_in_"):
        model_features = [str(feature) for feature in model.feature_names_in_]

        if set(model_features) != set(selected_features):
            missing_from_dataset = [
                feature for feature in model_features
                if feature not in X.columns
            ]

            extra_in_saved_list = [
                feature for feature in selected_features
                if feature not in model_features
            ]

            raise ValueError(
                "Saved feature list does not match the model's expected "
                "features.\n"
                f"Missing from dataset: {missing_from_dataset}\n"
                f"Extra in saved list: {extra_in_saved_list}"
            )

        X = X[model_features]
        selected_features = model_features

    print(f"\nPrediction feature matrix shape: {X.shape}")

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------
    print_section("GENERATING PREDICTIONS")

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "The loaded model does not support predict_proba(). "
            "A probabilistic classifier is required to create churn-risk scores."
        )

    probability_matrix = model.predict_proba(X)

    if probability_matrix.ndim != 2 or probability_matrix.shape[1] < 2:
        raise ValueError(
            "Unexpected predict_proba output shape: "
            f"{probability_matrix.shape}"
        )

    churn_probabilities = probability_matrix[:, 1]

    predicted_churn = (
        churn_probabilities >= PREDICTION_THRESHOLD
    ).astype(int)

    risk_levels = [
        assign_risk_level(float(probability))
        for probability in churn_probabilities
    ]

    # --------------------------------------------------------
    # Build dashboard-ready output
    # --------------------------------------------------------
    output_df = pd.DataFrame({
        CUSTOMER_ID_COLUMN: customer_ids.values,
        "ChurnProbability": churn_probabilities,
        "PredictedChurn": predicted_churn,
        "RiskLevel": risk_levels,
    })

    # Include the actual churn label when available.
    # This can help the dashboard compare predicted and actual churn.
    if TARGET_COLUMN in df.columns:
        output_df["ActualChurn"] = df[TARGET_COLUMN].values

    # Add selected features for dashboard filters and charts.
    for feature in selected_features:
        if feature not in output_df.columns:
            output_df[feature] = df[feature].values

    # Add useful optional columns when present.
    optional_dashboard_columns = [
        "CustomerLifetime",
        "TotalQuantity",
        "AvgOrderValue",
        "EngagementScore",
        "CustomerValueScore",
        "Recency",
        "Frequency",
        "Monetary",
        "PurchaseFrequency",
        "DaysPerOrder",
    ]

    for column in optional_dashboard_columns:
        if column in df.columns and column not in output_df.columns:
            output_df[column] = df[column].values

    # Rank highest-risk customers first.
    output_df = output_df.sort_values(
        by="ChurnProbability",
        ascending=False
    ).reset_index(drop=True)

    output_df.insert(
        0,
        "RiskRank",
        range(1, len(output_df) + 1)
    )

    # Mark top 20% highest-risk customers.
    top_20_count = max(1, int(np.ceil(len(output_df) * 0.20)))
    output_df["Top20PercentRisk"] = False
    output_df.loc[
        output_df.index[:top_20_count],
        "Top20PercentRisk"
    ] = True

    # Round probabilities for cleaner dashboard display.
    output_df["ChurnProbability"] = output_df[
        "ChurnProbability"
    ].round(6)

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    # --------------------------------------------------------
    # Display summary
    # --------------------------------------------------------
    print_section("EXPORT COMPLETED")

    high_risk_count = int(
        (output_df["RiskLevel"] == "High").sum()
    )
    medium_risk_count = int(
        (output_df["RiskLevel"] == "Medium").sum()
    )
    low_risk_count = int(
        (output_df["RiskLevel"] == "Low").sum()
    )
    predicted_churn_count = int(
        output_df["PredictedChurn"].sum()
    )

    print(f"Output path              : {OUTPUT_PATH.resolve()}")
    print(f"Total customers          : {len(output_df)}")
    print(f"Predicted churn customers: {predicted_churn_count}")
    print(f"High-risk customers      : {high_risk_count}")
    print(f"Medium-risk customers    : {medium_risk_count}")
    print(f"Low-risk customers       : {low_risk_count}")
    print(f"Top 20% customer count   : {top_20_count}")

    if TARGET_COLUMN in df.columns:
        actual_top_20 = output_df.loc[
            output_df["Top20PercentRisk"],
            "ActualChurn"
        ]

        if len(actual_top_20) > 0:
            precision_at_top_20 = actual_top_20.mean()

            print(
                "Precision@Top20%         : "
                f"{precision_at_top_20:.4f}"
            )

    print("\nPreview of dashboard-ready output:")
    preview_columns = [
        "RiskRank",
        CUSTOMER_ID_COLUMN,
        "ChurnProbability",
        "PredictedChurn",
        "RiskLevel",
        "Top20PercentRisk",
    ]

    if "ActualChurn" in output_df.columns:
        preview_columns.append("ActualChurn")

    print(output_df[preview_columns].head(10).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print_section("EXPORT FAILED")
        print(f"{type(error).__name__}: {error}")
        raise