# ============================================================
# RetailPulse
# F04 Customer Churn Prediction
# Final Model Evaluation
# ============================================================

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
TOP_PERCENTAGE = 0.20

TARGET = "Churn"
IDENTIFIER = "CustomerID"

# Models to compare.
MODEL_A_PATH = "models/churn/best_xgboost.pkl"

MODEL_B_PATH = (
    "models/churn/tuned_selected_xgboost.pkl"
)

# The selected dataset is used as the reference because the
# selected-feature tuning model was trained from this file.
REFERENCE_DATA_PATH = (
    "data/processed/selected_churn_features.csv"
)

# Additional datasets are used to recover features required
# by models trained on other feature configurations.
BASE_DATA_PATH = (
    "data/processed/churn_features.csv"
)

ADVANCED_DATA_PATH = (
    "data/processed/advanced_churn_features.csv"
)

# Output paths used in Parts 2 and 3.
REPORT_DIRECTORY = "outputs/reports"
FIGURE_DIRECTORY = "outputs/figures"
MODEL_DIRECTORY = "models/churn"

FINAL_REPORT_PATH = (
    "outputs/reports/final_churn_evaluation.txt"
)

METRICS_PATH = (
    "outputs/reports/final_model_comparison.csv"
)

PREDICTIONS_PATH = (
    "outputs/reports/final_churn_predictions.csv"
)

TOP_RISK_PATH = (
    "outputs/reports/final_top_20_percent_customers.csv"
)

ROC_CURVE_PATH = (
    "outputs/figures/final_model_roc_comparison.png"
)

CONFUSION_MATRIX_A_PATH = (
    "outputs/figures/final_model_a_confusion_matrix.png"
)

CONFUSION_MATRIX_B_PATH = (
    "outputs/figures/final_model_b_confusion_matrix.png"
)

# ============================================================
# Known Feature Configurations
# ============================================================

BASE_FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "TotalQuantity",
    "AvgOrderValue",
    "CustomerLifetime",
]

SELECTED_FEATURES = [
    "EngagementScore",
    "CustomerValueScore",
    "Recency",
    "FrequencyQuantity",
    "Frequency",
    "DaysPerOrder",
    "DailyOrders",
    "LoyaltyScore",
    "PurchaseFrequency",
    "TotalQuantity",
    "Monetary",
    "DailyRevenue",
    "LogMonetary",
    "QuantityEfficiency",
    "AvgOrderValue",
]

ADVANCED_FEATURES = [
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

# ============================================================
# Create Required Directories
# ============================================================

os.makedirs(
    REPORT_DIRECTORY,
    exist_ok=True,
)

os.makedirs(
    FIGURE_DIRECTORY,
    exist_ok=True,
)

os.makedirs(
    MODEL_DIRECTORY,
    exist_ok=True,
)

print("=" * 70)
print("FINAL CUSTOMER CHURN MODEL EVALUATION")
print("=" * 70)

# ============================================================
# Dataset Loading Function
# ============================================================

def load_dataset(
    path,
    dataset_name,
    required=True,
):
    """
    Load and validate a churn dataset.

    Parameters
    ----------
    path : str
        Location of the CSV file.

    dataset_name : str
        Display name used in console messages.

    required : bool
        Whether a missing file should stop execution.

    Returns
    -------
    pandas.DataFrame or None
        Validated dataset when available.
    """

    if not os.path.exists(path):

        if required:
            raise FileNotFoundError(
                f"{dataset_name} not found:\n"
                f"{path}"
            )

        print(
            f"Optional dataset not found: "
            f"{path}"
        )

        return None

    dataset = pd.read_csv(path)

    if dataset.empty:
        raise ValueError(
            f"{dataset_name} is empty."
        )

    required_columns = [
        IDENTIFIER,
        TARGET,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing columns: "
            f"{missing_columns}"
        )

    # Standardize customer identifiers to avoid merge problems.
    dataset[IDENTIFIER] = (
        dataset[IDENTIFIER]
        .astype(str)
        .str.strip()
    )

    dataset[TARGET] = (
        dataset[TARGET]
        .astype(int)
    )

    if dataset[TARGET].nunique() != 2:
        raise ValueError(
            f"{dataset_name} target must contain "
            "exactly two classes."
        )

    duplicate_count = dataset[
        IDENTIFIER
    ].duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(
            f"{dataset_name} contains "
            f"{duplicate_count} duplicate CustomerID values."
        )

    print(
        f"{dataset_name:<28}: "
        f"{dataset.shape}"
    )

    return dataset


# ============================================================
# Load Available Datasets
# ============================================================

print("\nLoading datasets...")

reference_df = load_dataset(
    REFERENCE_DATA_PATH,
    "Selected Feature Dataset",
    required=True,
)

base_df = load_dataset(
    BASE_DATA_PATH,
    "Base Feature Dataset",
    required=False,
)

advanced_df = load_dataset(
    ADVANCED_DATA_PATH,
    "Advanced Feature Dataset",
    required=False,
)

# ============================================================
# Model Loading Function
# ============================================================

def load_model(
    path,
    model_name,
):
    """
    Load a saved machine-learning model.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{model_name} not found:\n"
            f"{path}"
        )

    loaded_model = joblib.load(path)

    if not hasattr(
        loaded_model,
        "predict_proba",
    ):
        raise TypeError(
            f"{model_name} does not support "
            "predict_proba()."
        )

    print(
        f"{model_name:<28}: loaded"
    )

    return loaded_model


# ============================================================
# Load Models
# ============================================================

print("\nLoading saved models...")

model_a = load_model(
    MODEL_A_PATH,
    "Model A",
)

model_b = load_model(
    MODEL_B_PATH,
    "Model B",
)

# ============================================================
# Extract Model Feature Names
# ============================================================

def extract_model_features(model):
    """
    Retrieve feature names stored inside a fitted model.

    This supports both scikit-learn feature_names_in_
    and XGBoost Booster feature names.
    """

    if hasattr(
        model,
        "feature_names_in_",
    ):
        names = list(
            model.feature_names_in_
        )

        if names:
            return names

    if hasattr(
        model,
        "get_booster",
    ):
        booster = model.get_booster()

        if booster.feature_names:
            return list(
                booster.feature_names
            )

    return None


def get_model_feature_count(model):
    """
    Get the number of features expected by a model.
    """

    if hasattr(
        model,
        "n_features_in_",
    ):
        return int(
            model.n_features_in_
        )

    extracted_features = (
        extract_model_features(model)
    )

    if extracted_features:
        return len(extracted_features)

    return None


# ============================================================
# Resolve Model Features
# ============================================================

def resolve_model_features(
    model,
    model_name,
    preferred_features,
):
    """
    Determine the correct feature order for prediction.

    Stored feature names are preferred. When unavailable,
    the function uses known configurations based on the
    model's expected feature count.
    """

    stored_features = (
        extract_model_features(model)
    )

    if stored_features:
        print(
            f"{model_name} feature names "
            "were recovered from the model."
        )

        return stored_features

    feature_count = (
        get_model_feature_count(model)
    )

    if feature_count == len(
        preferred_features
    ):
        print(
            f"{model_name} feature names were not stored. "
            "Using the configured feature order."
        )

        return preferred_features

    if feature_count == len(
        BASE_FEATURES
    ):
        print(
            f"{model_name} detected as a "
            "six-feature model."
        )

        return BASE_FEATURES

    if feature_count == len(
        SELECTED_FEATURES
    ):
        print(
            f"{model_name} detected as a "
            "15-feature model."
        )

        return SELECTED_FEATURES

    if feature_count == len(
        ADVANCED_FEATURES
    ):
        print(
            f"{model_name} detected as a "
            "31-feature model."
        )

        return ADVANCED_FEATURES

    raise ValueError(
        f"Unable to determine feature order for "
        f"{model_name}. Expected feature count: "
        f"{feature_count}"
    )


print("\nResolving model feature configurations...")

model_a_features = resolve_model_features(
    model=model_a,
    model_name="Model A",
    preferred_features=BASE_FEATURES,
)

model_b_features = resolve_model_features(
    model=model_b,
    model_name="Model B",
    preferred_features=SELECTED_FEATURES,
)

print(
    f"\nModel A Feature Count : "
    f"{len(model_a_features)}"
)

print(
    f"Model B Feature Count : "
    f"{len(model_b_features)}"
)

# Detect whether best_xgboost.pkl was overwritten.

if len(model_a_features) != len(
    BASE_FEATURES
):
    print("\nWARNING")
    print("-" * 70)

    print(
        "Model A is not the expected six-feature model."
    )

    print(
        f"It currently expects "
        f"{len(model_a_features)} features."
    )

    print(
        "This probably means best_xgboost.pkl was "
        "overwritten by a later training run."
    )

# ============================================================
# Build Master Evaluation Dataset
# ============================================================

# Preserve the exact row order of the selected-feature dataset.
master_df = reference_df.copy()

additional_datasets = [
    base_df,
    advanced_df,
]

for additional_df in additional_datasets:

    if additional_df is None:
        continue

    # Verify that churn labels agree for shared customers.
    target_check = master_df[
        [IDENTIFIER, TARGET]
    ].merge(
        additional_df[
            [IDENTIFIER, TARGET]
        ],
        on=IDENTIFIER,
        how="inner",
        suffixes=(
            "_reference",
            "_additional",
        ),
    )

    inconsistent_targets = target_check[
        target_check[
            f"{TARGET}_reference"
        ]
        != target_check[
            f"{TARGET}_additional"
        ]
    ]

    if not inconsistent_targets.empty:
        raise ValueError(
            "Churn labels are inconsistent across "
            "the processed datasets."
        )

    new_feature_columns = [
        column
        for column in additional_df.columns
        if column not in master_df.columns
        and column not in [
            TARGET,
        ]
    ]

    if new_feature_columns:
        master_df = master_df.merge(
            additional_df[
                [IDENTIFIER]
                + new_feature_columns
            ],
            on=IDENTIFIER,
            how="left",
            validate="one_to_one",
        )

# ============================================================
# Validate Required Model Features
# ============================================================

all_required_features = list(
    dict.fromkeys(
        model_a_features
        + model_b_features
    )
)

missing_required_features = [
    feature
    for feature in all_required_features
    if feature not in master_df.columns
]

if missing_required_features:
    raise ValueError(
        "The following model features are missing "
        "from the available datasets:\n"
        f"{missing_required_features}"
    )

# ============================================================
# Clean Evaluation Data
# ============================================================

evaluation_columns = (
    [IDENTIFIER, TARGET]
    + all_required_features
)

evaluation_df = master_df[
    evaluation_columns
].copy()

evaluation_df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True,
)

missing_before = (
    evaluation_df[
        all_required_features
    ]
    .isna()
    .sum()
    .sum()
)

if missing_before > 0:
    print(
        f"\nMissing feature values found: "
        f"{missing_before}"
    )

    print(
        "Filling missing feature values "
        "with column medians..."
    )

    for feature in all_required_features:
        if evaluation_df[
            feature
        ].isna().any():

            median_value = evaluation_df[
                feature
            ].median()

            if pd.isna(median_value):
                raise ValueError(
                    f"Feature '{feature}' contains "
                    "only missing values."
                )

            evaluation_df[
                feature
            ] = evaluation_df[
                feature
            ].fillna(
                median_value
            )

# ============================================================
# Create Shared Train/Test Split
# ============================================================

# Both models must be evaluated on exactly the same customers.

row_indices = np.arange(
    len(evaluation_df)
)

train_indices, test_indices = (
    train_test_split(
        row_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=evaluation_df[TARGET],
    )
)

test_df = evaluation_df.iloc[
    test_indices
].copy()

test_df.reset_index(
    drop=True,
    inplace=True,
)

y_test = test_df[
    TARGET
].astype(int)

X_test_model_a = test_df[
    model_a_features
].copy()

X_test_model_b = test_df[
    model_b_features
].copy()

# ============================================================
# Final Validation
# ============================================================

if len(X_test_model_a) != len(
    X_test_model_b
):
    raise ValueError(
        "The models do not have the same "
        "number of test samples."
    )

if not X_test_model_a.index.equals(
    X_test_model_b.index
):
    raise ValueError(
        "The model test-set indices do not match."
    )

print("\n" + "=" * 70)
print("SHARED EVALUATION DATA")
print("=" * 70)

print(
    f"Total Customers       : "
    f"{len(evaluation_df)}"
)

print(
    f"Training Customers    : "
    f"{len(train_indices)}"
)

print(
    f"Testing Customers     : "
    f"{len(test_df)}"
)

print(
    f"Non-Churn Test Cases  : "
    f"{(y_test == 0).sum()}"
)

print(
    f"Churn Test Cases      : "
    f"{(y_test == 1).sum()}"
)

print(
    f"Top Percentage        : "
    f"{TOP_PERCENTAGE:.0%}"
)

print(
    f"Top Customer Count    : "
    f"{int(np.ceil(len(test_df) * TOP_PERCENTAGE))}"
)

print("\nModel A Features")

for feature in model_a_features:
    print(f"✔ {feature}")

print("\nModel B Features")

for feature in model_b_features:
    print(f"✔ {feature}")

print(
    "\nData preparation completed successfully."
)

print(
    "Proceeding to predictions and "
    "Precision@Top 20% evaluation..."
)
# ============================================================
# Precision-at-Top-Percentage Function
# ============================================================

def calculate_top_percentage_metrics(
    y_true,
    probabilities,
    top_percentage=0.20,
):
    """
    Calculate ranking metrics for the highest-risk customers.

    Customers are sorted by predicted churn probability.
    The top percentage represents the customers targeted
    by a retention campaign.
    """

    y_true_array = np.asarray(
        y_true,
        dtype=int,
    )

    probability_array = np.asarray(
        probabilities,
        dtype=float,
    )

    if len(y_true_array) != len(
        probability_array
    ):
        raise ValueError(
            "Target and probability arrays must "
            "have the same length."
        )

    if not 0 < top_percentage <= 1:
        raise ValueError(
            "top_percentage must be between 0 and 1."
        )

    total_customers = len(
        y_true_array
    )

    top_customer_count = int(
        np.ceil(
            total_customers
            * top_percentage
        )
    )

    ranked_indices = np.argsort(
        probability_array
    )[::-1]

    top_indices = ranked_indices[
        :top_customer_count
    ]

    top_actual_values = y_true_array[
        top_indices
    ]

    churners_in_top_group = int(
        top_actual_values.sum()
    )

    total_actual_churners = int(
        y_true_array.sum()
    )

    overall_churn_rate = float(
        y_true_array.mean()
    )

    precision_at_top = (
        churners_in_top_group
        / top_customer_count
        if top_customer_count > 0
        else 0.0
    )

    recall_at_top = (
        churners_in_top_group
        / total_actual_churners
        if total_actual_churners > 0
        else 0.0
    )

    lift_at_top = (
        precision_at_top
        / overall_churn_rate
        if overall_churn_rate > 0
        else 0.0
    )

    return {
        "TopPercentage": top_percentage,
        "TopCustomerCount": top_customer_count,
        "ChurnersInTopGroup": churners_in_top_group,
        "TotalActualChurners": total_actual_churners,
        "OverallChurnRate": overall_churn_rate,
        "PrecisionAtTop20": precision_at_top,
        "RecallAtTop20": recall_at_top,
        "LiftAtTop20": lift_at_top,
        "TopIndices": top_indices,
    }


# ============================================================
# Model Evaluation Function
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name,
):
    """
    Generate standard classification metrics and
    top-20-percent ranking metrics.
    """

    print(
        f"\nGenerating predictions for "
        f"{model_name}..."
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    standard_metrics = {
        "Accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "F1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "ROC-AUC": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    ranking_metrics = (
        calculate_top_percentage_metrics(
            y_true=y_test,
            probabilities=probabilities,
            top_percentage=TOP_PERCENTAGE,
        )
    )

    combined_metrics = {
        **standard_metrics,
        "Precision@Top20": ranking_metrics[
            "PrecisionAtTop20"
        ],
        "Recall@Top20": ranking_metrics[
            "RecallAtTop20"
        ],
        "Lift@Top20": ranking_metrics[
            "LiftAtTop20"
        ],
        "TopCustomerCount": ranking_metrics[
            "TopCustomerCount"
        ],
        "ChurnersInTopGroup": ranking_metrics[
            "ChurnersInTopGroup"
        ],
    }

    model_confusion_matrix = (
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    model_classification_report = (
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    print(
        f"{model_name} predictions completed."
    )

    return {
        "ModelName": model_name,
        "Predictions": predictions,
        "Probabilities": probabilities,
        "Metrics": combined_metrics,
        "RankingMetrics": ranking_metrics,
        "ConfusionMatrix": model_confusion_matrix,
        "ClassificationReport": (
            model_classification_report
        ),
    }


# ============================================================
# Evaluate Both Models
# ============================================================

model_a_result = evaluate_model(
    model=model_a,
    X_test=X_test_model_a,
    y_test=y_test,
    model_name="Model A",
)

model_b_result = evaluate_model(
    model=model_b,
    X_test=X_test_model_b,
    y_test=y_test,
    model_name="Model B",
)

# ============================================================
# Display Model A Performance
# ============================================================

print("\n" + "=" * 70)
print("MODEL A PERFORMANCE")
print("=" * 70)

for metric_name, metric_value in (
    model_a_result["Metrics"].items()
):
    if metric_name in [
        "TopCustomerCount",
        "ChurnersInTopGroup",
    ]:
        print(
            f"{metric_name:<24}: "
            f"{int(metric_value)}"
        )
    else:
        print(
            f"{metric_name:<24}: "
            f"{metric_value:.4f}"
        )

print("\nModel A Classification Report")
print("-" * 70)

print(
    model_a_result[
        "ClassificationReport"
    ]
)

# ============================================================
# Display Model B Performance
# ============================================================

print("\n" + "=" * 70)
print("MODEL B PERFORMANCE")
print("=" * 70)

for metric_name, metric_value in (
    model_b_result["Metrics"].items()
):
    if metric_name in [
        "TopCustomerCount",
        "ChurnersInTopGroup",
    ]:
        print(
            f"{metric_name:<24}: "
            f"{int(metric_value)}"
        )
    else:
        print(
            f"{metric_name:<24}: "
            f"{metric_value:.4f}"
        )

print("\nModel B Classification Report")
print("-" * 70)

print(
    model_b_result[
        "ClassificationReport"
    ]
)

# ============================================================
# Create Model Comparison Table
# ============================================================

comparison_metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC",
    "Precision@Top20",
    "Recall@Top20",
    "Lift@Top20",
]

comparison_rows = []

for metric_name in comparison_metric_names:

    model_a_value = model_a_result[
        "Metrics"
    ][metric_name]

    model_b_value = model_b_result[
        "Metrics"
    ][metric_name]

    difference = (
        model_b_value
        - model_a_value
    )

    comparison_rows.append(
        {
            "Metric": metric_name,
            "ModelA": model_a_value,
            "ModelB": model_b_value,
            "Difference_B_minus_A": (
                difference
            ),
        }
    )

comparison_df = pd.DataFrame(
    comparison_rows
)

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

for _, row in comparison_df.iterrows():

    print(
        f"{row['Metric']:<20} "
        f"Model A: {row['ModelA']:.4f} | "
        f"Model B: {row['ModelB']:.4f} | "
        f"Difference: "
        f"{row['Difference_B_minus_A']:+.4f}"
    )

# ============================================================
# Build Customer-Level Prediction Table
# ============================================================

prediction_results = test_df[
    [
        IDENTIFIER,
        TARGET,
    ]
].copy()

prediction_results.rename(
    columns={
        TARGET: "ActualChurn",
    },
    inplace=True,
)

prediction_results[
    "ModelAProbability"
] = model_a_result[
    "Probabilities"
]

prediction_results[
    "ModelAPrediction"
] = model_a_result[
    "Predictions"
]

prediction_results[
    "ModelBProbability"
] = model_b_result[
    "Probabilities"
]

prediction_results[
    "ModelBPrediction"
] = model_b_result[
    "Predictions"
]

# Rank 1 represents the customer with the highest
# predicted churn probability.

prediction_results[
    "ModelARiskRank"
] = prediction_results[
    "ModelAProbability"
].rank(
    method="first",
    ascending=False,
).astype(int)

prediction_results[
    "ModelBRiskRank"
] = prediction_results[
    "ModelBProbability"
].rank(
    method="first",
    ascending=False,
).astype(int)

model_a_top_count = model_a_result[
    "RankingMetrics"
]["TopCustomerCount"]

model_b_top_count = model_b_result[
    "RankingMetrics"
]["TopCustomerCount"]

prediction_results[
    "ModelATop20Flag"
] = (
    prediction_results[
        "ModelARiskRank"
    ]
    <= model_a_top_count
).astype(int)

prediction_results[
    "ModelBTop20Flag"
] = (
    prediction_results[
        "ModelBRiskRank"
    ]
    <= model_b_top_count
).astype(int)

# ============================================================
# Create Top-Risk Customer Tables
# ============================================================

model_a_top_customers = (
    prediction_results[
        prediction_results[
            "ModelATop20Flag"
        ] == 1
    ]
    .sort_values(
        by="ModelAProbability",
        ascending=False,
    )
    .copy()
)

model_b_top_customers = (
    prediction_results[
        prediction_results[
            "ModelBTop20Flag"
        ] == 1
    ]
    .sort_values(
        by="ModelBProbability",
        ascending=False,
    )
    .copy()
)

print("\n" + "=" * 70)
print("TOP 20% CUSTOMER SUMMARY")
print("=" * 70)

print(
    f"Model A Top Customers       : "
    f"{len(model_a_top_customers)}"
)

print(
    f"Model A Actual Churners     : "
    f"{model_a_top_customers['ActualChurn'].sum()}"
)

print(
    f"Model A Precision@Top20     : "
    f"{model_a_result['Metrics']['Precision@Top20']:.4f}"
)

print(
    f"Model A Recall@Top20        : "
    f"{model_a_result['Metrics']['Recall@Top20']:.4f}"
)

print(
    f"Model A Lift@Top20          : "
    f"{model_a_result['Metrics']['Lift@Top20']:.4f}"
)

print()

print(
    f"Model B Top Customers       : "
    f"{len(model_b_top_customers)}"
)

print(
    f"Model B Actual Churners     : "
    f"{model_b_top_customers['ActualChurn'].sum()}"
)

print(
    f"Model B Precision@Top20     : "
    f"{model_b_result['Metrics']['Precision@Top20']:.4f}"
)

print(
    f"Model B Recall@Top20        : "
    f"{model_b_result['Metrics']['Recall@Top20']:.4f}"
)

print(
    f"Model B Lift@Top20          : "
    f"{model_b_result['Metrics']['Lift@Top20']:.4f}"
)

# ============================================================
# Acceptance Target Evaluation
# ============================================================

ROC_AUC_TARGET = 0.88
PRECISION_TOP20_TARGET = 0.75

model_a_target_status = {
    "ROCAUCTargetMet": (
        model_a_result[
            "Metrics"
        ]["ROC-AUC"]
        >= ROC_AUC_TARGET
    ),
    "PrecisionTop20TargetMet": (
        model_a_result[
            "Metrics"
        ]["Precision@Top20"]
        >= PRECISION_TOP20_TARGET
    ),
}

model_b_target_status = {
    "ROCAUCTargetMet": (
        model_b_result[
            "Metrics"
        ]["ROC-AUC"]
        >= ROC_AUC_TARGET
    ),
    "PrecisionTop20TargetMet": (
        model_b_result[
            "Metrics"
        ]["Precision@Top20"]
        >= PRECISION_TOP20_TARGET
    ),
}

print("\n" + "=" * 70)
print("COMPANY ACCEPTANCE TARGETS")
print("=" * 70)

print(
    f"Required ROC-AUC            : "
    f">= {ROC_AUC_TARGET:.2f}"
)

print(
    f"Required Precision@Top20    : "
    f">= {PRECISION_TOP20_TARGET:.2f}"
)

print("\nModel A")

print(
    "ROC-AUC Target             : "
    + (
        "ACHIEVED"
        if model_a_target_status[
            "ROCAUCTargetMet"
        ]
        else "NOT ACHIEVED"
    )
)

print(
    "Precision@Top20 Target     : "
    + (
        "ACHIEVED"
        if model_a_target_status[
            "PrecisionTop20TargetMet"
        ]
        else "NOT ACHIEVED"
    )
)

print("\nModel B")

print(
    "ROC-AUC Target             : "
    + (
        "ACHIEVED"
        if model_b_target_status[
            "ROCAUCTargetMet"
        ]
        else "NOT ACHIEVED"
    )
)

print(
    "Precision@Top20 Target     : "
    + (
        "ACHIEVED"
        if model_b_target_status[
            "PrecisionTop20TargetMet"
        ]
        else "NOT ACHIEVED"
    )
)

# ============================================================
# Save Confusion Matrix — Model A
# ============================================================

model_a_confusion_display = (
    ConfusionMatrixDisplay(
        confusion_matrix=model_a_result[
            "ConfusionMatrix"
        ],
        display_labels=[
            "Non-Churn",
            "Churn",
        ],
    )
)

model_a_confusion_display.plot(
    values_format="d",
)

plt.title(
    "Model A — Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_A_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    f"\nModel A confusion matrix saved: "
    f"{CONFUSION_MATRIX_A_PATH}"
)

# ============================================================
# Save Confusion Matrix — Model B
# ============================================================

model_b_confusion_display = (
    ConfusionMatrixDisplay(
        confusion_matrix=model_b_result[
            "ConfusionMatrix"
        ],
        display_labels=[
            "Non-Churn",
            "Churn",
        ],
    )
)

model_b_confusion_display.plot(
    values_format="d",
)

plt.title(
    "Model B — Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_B_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    f"Model B confusion matrix saved: "
    f"{CONFUSION_MATRIX_B_PATH}"
)

# ============================================================
# Save ROC Curve Comparison
# ============================================================

plt.figure(
    figsize=(9, 7)
)

axis = plt.gca()

RocCurveDisplay.from_predictions(
    y_test,
    model_a_result[
        "Probabilities"
    ],
    name=(
        "Model A "
        f"(AUC={model_a_result['Metrics']['ROC-AUC']:.4f})"
    ),
    ax=axis,
)

RocCurveDisplay.from_predictions(
    y_test,
    model_b_result[
        "Probabilities"
    ],
    name=(
        "Model B "
        f"(AUC={model_b_result['Metrics']['ROC-AUC']:.4f})"
    ),
    ax=axis,
)

plt.title(
    "Final Churn Model ROC Comparison"
)

plt.tight_layout()

plt.savefig(
    ROC_CURVE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    f"ROC comparison saved: "
    f"{ROC_CURVE_PATH}"
)

print(
    "\nPrediction and metric evaluation "
    "completed successfully."
)

print(
    "Proceeding to final model selection "
    "and report generation..."
)
# ============================================================
# Final Output Paths
# ============================================================

FINAL_MODEL_PATH = (
    "models/churn/final_churn_model.pkl"
)

FINAL_FEATURES_PATH = (
    "models/churn/final_churn_features.pkl"
)

MODEL_A_TOP_RISK_PATH = (
    "outputs/reports/model_a_top_20_percent_customers.csv"
)

MODEL_B_TOP_RISK_PATH = (
    "outputs/reports/model_b_top_20_percent_customers.csv"
)

# ============================================================
# Final Model-Selection Function
# ============================================================

def select_final_model(
    result_a,
    result_b,
):
    """
    Select the recommended churn model.

    Selection priority:
    1. ROC-AUC
    2. Precision@Top20
    3. F1 score
    4. Recall
    """

    metrics_a = result_a["Metrics"]
    metrics_b = result_b["Metrics"]

    selection_metrics = [
        "ROC-AUC",
        "Precision@Top20",
        "F1",
        "Recall",
    ]

    for metric_name in selection_metrics:

        value_a = metrics_a[
            metric_name
        ]

        value_b = metrics_b[
            metric_name
        ]

        if not np.isclose(
            value_a,
            value_b,
            atol=1e-6,
        ):
            if value_a > value_b:
                return "Model A", metric_name

            return "Model B", metric_name

    # Complete tie fallback.
    return "Model A", "Complete metric tie"


# ============================================================
# Select Recommended Model
# ============================================================

recommended_model_name, decision_metric = (
    select_final_model(
        model_a_result,
        model_b_result,
    )
)

if recommended_model_name == "Model A":

    recommended_model = model_a

    recommended_features = (
        model_a_features
    )

    recommended_result = (
        model_a_result
    )

    recommended_top_customers = (
        model_a_top_customers
    )

    recommended_target_status = (
        model_a_target_status
    )

    source_model_path = (
        MODEL_A_PATH
    )

else:

    recommended_model = model_b

    recommended_features = (
        model_b_features
    )

    recommended_result = (
        model_b_result
    )

    recommended_top_customers = (
        model_b_top_customers
    )

    recommended_target_status = (
        model_b_target_status
    )

    source_model_path = (
        MODEL_B_PATH
    )

print("\n" + "=" * 70)
print("FINAL MODEL RECOMMENDATION")
print("=" * 70)

print(
    f"Recommended Model          : "
    f"{recommended_model_name}"
)

print(
    f"Source Model Path          : "
    f"{source_model_path}"
)

print(
    f"Selection Metric           : "
    f"{decision_metric}"
)

print(
    f"Feature Count              : "
    f"{len(recommended_features)}"
)

print(
    f"ROC-AUC                    : "
    f"{recommended_result['Metrics']['ROC-AUC']:.4f}"
)

print(
    f"Precision@Top20            : "
    f"{recommended_result['Metrics']['Precision@Top20']:.4f}"
)

print(
    f"Recall@Top20               : "
    f"{recommended_result['Metrics']['Recall@Top20']:.4f}"
)

print(
    f"Lift@Top20                 : "
    f"{recommended_result['Metrics']['Lift@Top20']:.4f}"
)

# ============================================================
# Add Recommended Model Results
# ============================================================

if recommended_model_name == "Model A":

    prediction_results[
        "RecommendedProbability"
    ] = prediction_results[
        "ModelAProbability"
    ]

    prediction_results[
        "RecommendedPrediction"
    ] = prediction_results[
        "ModelAPrediction"
    ]

    prediction_results[
        "RecommendedRiskRank"
    ] = prediction_results[
        "ModelARiskRank"
    ]

    prediction_results[
        "RecommendedTop20Flag"
    ] = prediction_results[
        "ModelATop20Flag"
    ]

else:

    prediction_results[
        "RecommendedProbability"
    ] = prediction_results[
        "ModelBProbability"
    ]

    prediction_results[
        "RecommendedPrediction"
    ] = prediction_results[
        "ModelBPrediction"
    ]

    prediction_results[
        "RecommendedRiskRank"
    ] = prediction_results[
        "ModelBRiskRank"
    ]

    prediction_results[
        "RecommendedTop20Flag"
    ] = prediction_results[
        "ModelBTop20Flag"
    ]

prediction_results[
    "RecommendedModel"
] = recommended_model_name

prediction_results.sort_values(
    by="RecommendedProbability",
    ascending=False,
    inplace=True,
)

prediction_results.reset_index(
    drop=True,
    inplace=True,
)

# ============================================================
# Save Model Comparison Metrics
# ============================================================

comparison_df.to_csv(
    METRICS_PATH,
    index=False,
)

print(
    f"\nModel comparison saved: "
    f"{METRICS_PATH}"
)

# ============================================================
# Save Customer-Level Predictions
# ============================================================

prediction_results.to_csv(
    PREDICTIONS_PATH,
    index=False,
)

print(
    f"Prediction results saved: "
    f"{PREDICTIONS_PATH}"
)

# ============================================================
# Save Model A Top-20% Customers
# ============================================================

model_a_top_customers.to_csv(
    MODEL_A_TOP_RISK_PATH,
    index=False,
)

print(
    f"Model A top-risk customers saved: "
    f"{MODEL_A_TOP_RISK_PATH}"
)

# ============================================================
# Save Model B Top-20% Customers
# ============================================================

model_b_top_customers.to_csv(
    MODEL_B_TOP_RISK_PATH,
    index=False,
)

print(
    f"Model B top-risk customers saved: "
    f"{MODEL_B_TOP_RISK_PATH}"
)

# ============================================================
# Save Recommended Top-20% Customers
# ============================================================

recommended_top_customers = (
    prediction_results[
        prediction_results[
            "RecommendedTop20Flag"
        ] == 1
    ]
    .sort_values(
        by="RecommendedProbability",
        ascending=False,
    )
    .copy()
)

recommended_top_customers.to_csv(
    TOP_RISK_PATH,
    index=False,
)

print(
    f"Recommended top-risk customers saved: "
    f"{TOP_RISK_PATH}"
)

# ============================================================
# Save Final Recommended Model
# ============================================================

joblib.dump(
    recommended_model,
    FINAL_MODEL_PATH,
)

print(
    f"Final model saved: "
    f"{FINAL_MODEL_PATH}"
)

# ============================================================
# Save Final Model Feature Order
# ============================================================

joblib.dump(
    recommended_features,
    FINAL_FEATURES_PATH,
)

print(
    f"Final feature order saved: "
    f"{FINAL_FEATURES_PATH}"
)

# ============================================================
# Calculate Business Summary
# ============================================================

top_customer_count = (
    recommended_result[
        "Metrics"
    ]["TopCustomerCount"]
)

churners_in_top_group = (
    recommended_result[
        "Metrics"
    ]["ChurnersInTopGroup"]
)

non_churners_in_top_group = (
    top_customer_count
    - churners_in_top_group
)

total_test_churners = int(
    y_test.sum()
)

missed_churners = (
    total_test_churners
    - churners_in_top_group
)

overall_test_churn_rate = float(
    y_test.mean()
)

target_group_churn_rate = (
    recommended_result[
        "Metrics"
    ]["Precision@Top20"]
)

# ============================================================
# Write Complete Evaluation Report
# ============================================================

with open(
    FINAL_REPORT_PATH,
    "w",
    encoding="utf-8",
) as report:

    report.write("RetailPulse\n")

    report.write(
        "F04 Customer Churn Prediction\n"
    )

    report.write(
        "Final Model Evaluation Report\n"
    )

    report.write("=" * 75)
    report.write("\n\n")

    # --------------------------------------------------------
    # Evaluation information
    # --------------------------------------------------------

    report.write(
        "Evaluation Configuration\n"
    )

    report.write("-" * 75)
    report.write("\n")

    report.write(
        f"Random State: {RANDOM_STATE}\n"
    )

    report.write(
        f"Test Size: {TEST_SIZE:.0%}\n"
    )

    report.write(
        f"Top Customer Percentage: "
        f"{TOP_PERCENTAGE:.0%}\n"
    )

    report.write(
        f"Total Customers: "
        f"{len(evaluation_df)}\n"
    )

    report.write(
        f"Training Customers: "
        f"{len(train_indices)}\n"
    )

    report.write(
        f"Testing Customers: "
        f"{len(test_df)}\n"
    )

    report.write(
        f"Test Churners: "
        f"{total_test_churners}\n"
    )

    report.write(
        f"Test Non-Churners: "
        f"{int((y_test == 0).sum())}\n\n"
    )

    # --------------------------------------------------------
    # Model configurations
    # --------------------------------------------------------

    report.write(
        "Model Configurations\n"
    )

    report.write("-" * 75)
    report.write("\n")

    report.write(
        f"Model A Path: {MODEL_A_PATH}\n"
    )

    report.write(
        f"Model A Feature Count: "
        f"{len(model_a_features)}\n"
    )

    report.write(
        "Model A Features:\n"
    )

    for feature in model_a_features:
        report.write(
            f"  - {feature}\n"
        )

    report.write("\n")

    report.write(
        f"Model B Path: {MODEL_B_PATH}\n"
    )

    report.write(
        f"Model B Feature Count: "
        f"{len(model_b_features)}\n"
    )

    report.write(
        "Model B Features:\n"
    )

    for feature in model_b_features:
        report.write(
            f"  - {feature}\n"
        )

    report.write("\n")

    # --------------------------------------------------------
    # Model A metrics
    # --------------------------------------------------------

    report.write(
        "Model A Performance\n"
    )

    report.write("-" * 75)
    report.write("\n")

    for metric_name, metric_value in (
        model_a_result[
            "Metrics"
        ].items()
    ):

        if metric_name in [
            "TopCustomerCount",
            "ChurnersInTopGroup",
        ]:
            report.write(
                f"{metric_name}: "
                f"{int(metric_value)}\n"
            )

        else:
            report.write(
                f"{metric_name}: "
                f"{metric_value:.4f}\n"
            )

    report.write("\n")

    report.write(
        "Model A Classification Report\n"
    )

    report.write(
        model_a_result[
            "ClassificationReport"
        ]
    )

    report.write("\n")

    report.write(
        "Model A Confusion Matrix\n"
    )

    report.write(
        np.array2string(
            model_a_result[
                "ConfusionMatrix"
            ]
        )
    )

    report.write("\n\n")

    # --------------------------------------------------------
    # Model B metrics
    # --------------------------------------------------------

    report.write(
        "Model B Performance\n"
    )

    report.write("-" * 75)
    report.write("\n")

    for metric_name, metric_value in (
        model_b_result[
            "Metrics"
        ].items()
    ):

        if metric_name in [
            "TopCustomerCount",
            "ChurnersInTopGroup",
        ]:
            report.write(
                f"{metric_name}: "
                f"{int(metric_value)}\n"
            )

        else:
            report.write(
                f"{metric_name}: "
                f"{metric_value:.4f}\n"
            )

    report.write("\n")

    report.write(
        "Model B Classification Report\n"
    )

    report.write(
        model_b_result[
            "ClassificationReport"
        ]
    )

    report.write("\n")

    report.write(
        "Model B Confusion Matrix\n"
    )

    report.write(
        np.array2string(
            model_b_result[
                "ConfusionMatrix"
            ]
        )
    )

    report.write("\n\n")

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    report.write(
        "Model Comparison\n"
    )

    report.write("-" * 75)
    report.write("\n")

    report.write(
        comparison_df.to_string(
            index=False,
        )
    )

    report.write("\n\n")

    # --------------------------------------------------------
    # Business ranking metrics
    # --------------------------------------------------------

    report.write(
        "Recommended Model Business Ranking Summary\n"
    )

    report.write("-" * 75)
    report.write("\n")

    report.write(
        f"Recommended Model: "
        f"{recommended_model_name}\n"
    )

    report.write(
        f"Customers in Top 20%: "
        f"{top_customer_count}\n"
    )

    report.write(
        f"Actual Churners in Top 20%: "
        f"{churners_in_top_group}\n"
    )

    report.write(
        f"Non-Churners in Top 20%: "
        f"{non_churners_in_top_group}\n"
    )

    report.write(
        f"Total Test Churners: "
        f"{total_test_churners}\n"
    )

    report.write(
        f"Churners Outside Top 20%: "
        f"{missed_churners}\n"
    )

    report.write(
        f"Overall Test Churn Rate: "
        f"{overall_test_churn_rate:.4f}\n"
    )

    report.write(
        f"Top-20% Churn Rate: "
        f"{target_group_churn_rate:.4f}\n"
    )

    report.write(
        f"Precision@Top20: "
        f"{recommended_result['Metrics']['Precision@Top20']:.4f}\n"
    )

    report.write(
        f"Recall@Top20: "
        f"{recommended_result['Metrics']['Recall@Top20']:.4f}\n"
    )

    report.write(
        f"Lift@Top20: "
        f"{recommended_result['Metrics']['Lift@Top20']:.4f}\n\n"
    )

    # --------------------------------------------------------
    # Acceptance targets
    # --------------------------------------------------------

    report.write(
        "Company Acceptance Targets\n"
    )

    report.write("-" * 75)
    report.write("\n")

    report.write(
        f"Required ROC-AUC: "
        f">= {ROC_AUC_TARGET:.2f}\n"
    )

    report.write(
        f"Recommended ROC-AUC: "
        f"{recommended_result['Metrics']['ROC-AUC']:.4f}\n"
    )

    report.write(
        "ROC-AUC Target Status: "
    )

    report.write(
        "ACHIEVED\n"
        if recommended_target_status[
            "ROCAUCTargetMet"
        ]
        else "NOT ACHIEVED\n"
    )

    report.write(
        f"\nRequired Precision@Top20: "
        f">= {PRECISION_TOP20_TARGET:.2f}\n"
    )

    report.write(
        f"Recommended Precision@Top20: "
        f"{recommended_result['Metrics']['Precision@Top20']:.4f}\n"
    )

    report.write(
        "Precision@Top20 Target Status: "
    )

    report.write(
        "ACHIEVED\n"
        if recommended_target_status[
            "PrecisionTop20TargetMet"
        ]
        else "NOT ACHIEVED\n"
    )

    both_targets_met = (
        recommended_target_status[
            "ROCAUCTargetMet"
        ]
        and recommended_target_status[
            "PrecisionTop20TargetMet"
        ]
    )

    report.write(
        "\nOverall Acceptance Status: "
    )

    report.write(
        "ALL TARGETS ACHIEVED\n"
        if both_targets_met
        else "TARGETS PARTIALLY ACHIEVED "
        "OR NOT ACHIEVED\n"
    )

    report.write("\n")

    # --------------------------------------------------------
    # Final recommendation
    # --------------------------------------------------------

    report.write(
        "Final Recommendation\n"
    )

    report.write("-" * 75)
    report.write("\n")

    report.write(
        f"Recommended Model: "
        f"{recommended_model_name}\n"
    )

    report.write(
        f"Selection Metric: "
        f"{decision_metric}\n"
    )

    report.write(
        f"Source Model: "
        f"{source_model_path}\n"
    )

    report.write(
        f"Final Saved Model: "
        f"{FINAL_MODEL_PATH}\n"
    )

    report.write(
        f"Final Feature Count: "
        f"{len(recommended_features)}\n"
    )

    report.write(
        "The recommended model was selected "
        "primarily using ROC-AUC, followed by "
        "Precision@Top20, F1 score, and recall "
        "as tie-breakers.\n"
    )

print(
    f"Final evaluation report saved: "
    f"{FINAL_REPORT_PATH}"
)

# ============================================================
# Display Recommended Features
# ============================================================

print("\nRecommended Model Features")
print("-" * 70)

for position, feature in enumerate(
    recommended_features,
    start=1,
):
    print(
        f"{position:>2}. {feature}"
    )

# ============================================================
# Final Acceptance Summary
# ============================================================

print("\n" + "=" * 70)
print("F04 ACCEPTANCE SUMMARY")
print("=" * 70)

print(
    f"ROC-AUC Target             : "
    f">= {ROC_AUC_TARGET:.2f}"
)

print(
    f"Final ROC-AUC              : "
    f"{recommended_result['Metrics']['ROC-AUC']:.4f}"
)

print(
    "ROC-AUC Status            : "
    + (
        "ACHIEVED"
        if recommended_target_status[
            "ROCAUCTargetMet"
        ]
        else "NOT ACHIEVED"
    )
)

print()

print(
    f"Precision@Top20 Target     : "
    f">= {PRECISION_TOP20_TARGET:.2f}"
)

print(
    f"Final Precision@Top20      : "
    f"{recommended_result['Metrics']['Precision@Top20']:.4f}"
)

print(
    "Precision@Top20 Status    : "
    + (
        "ACHIEVED"
        if recommended_target_status[
            "PrecisionTop20TargetMet"
        ]
        else "NOT ACHIEVED"
    )
)

# ============================================================
# Generated Files
# ============================================================

generated_files = [
    FINAL_MODEL_PATH,
    FINAL_FEATURES_PATH,
    FINAL_REPORT_PATH,
    METRICS_PATH,
    PREDICTIONS_PATH,
    TOP_RISK_PATH,
    MODEL_A_TOP_RISK_PATH,
    MODEL_B_TOP_RISK_PATH,
    ROC_CURVE_PATH,
    CONFUSION_MATRIX_A_PATH,
    CONFUSION_MATRIX_B_PATH,
]

print("\nGenerated Files")
print("-" * 70)

for generated_file in generated_files:
    print(
        f"✔ {generated_file}"
    )

# ============================================================
# Final Status
# ============================================================

both_targets_met = (
    recommended_target_status[
        "ROCAUCTargetMet"
    ]
    and recommended_target_status[
        "PrecisionTop20TargetMet"
    ]
)

print("\n" + "=" * 70)

if both_targets_met:

    print(
        "F04 CUSTOMER CHURN ACCEPTANCE "
        "TARGETS ACHIEVED"
    )

elif recommended_target_status[
    "PrecisionTop20TargetMet"
]:

    print(
        "PRECISION@TOP20 TARGET ACHIEVED"
    )

    print(
        "ROC-AUC TARGET REQUIRES "
        "FURTHER IMPROVEMENT"
    )

elif recommended_target_status[
    "ROCAUCTargetMet"
]:

    print(
        "ROC-AUC TARGET ACHIEVED"
    )

    print(
        "PRECISION@TOP20 TARGET REQUIRES "
        "FURTHER IMPROVEMENT"
    )

else:

    print(
        "THE F04 PIPELINE IS COMPLETE"
    )

    print(
        "THE COMPANY ACCEPTANCE TARGETS "
        "HAVE NOT YET BEEN ACHIEVED"
    )

print("=" * 70)

print(
    "\nFINAL CUSTOMER CHURN MODEL "
    "EVALUATION COMPLETED SUCCESSFULLY"
)

print("=" * 70)