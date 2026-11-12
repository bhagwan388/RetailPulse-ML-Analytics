# ============================================================
# RetailPulse
# F04 Customer Churn Prediction
# SHAP Model Explainability
# ============================================================

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

MODEL_PATH = (
    "models/churn/final_churn_model.pkl"
)

FEATURES_PATH = (
    "models/churn/final_churn_features.pkl"
)

DATA_PATH = (
    "data/processed/churn_features.csv"
)

TARGET = "Churn"
IDENTIFIER = "CustomerID"

# Number of customers used for global SHAP analysis.
# Using a sample keeps execution fast while preserving
# a representative view of model behaviour.
SHAP_SAMPLE_SIZE = 1000

# Number of highest-risk customers for individual explanations.
TOP_CUSTOMER_COUNT = 20

# Output directories
REPORT_DIRECTORY = "outputs/reports"
FIGURE_DIRECTORY = "outputs/figures"

# Output files created in Parts 2 and 3.
SHAP_IMPORTANCE_PATH = (
    "outputs/reports/shap_feature_importance.csv"
)

CUSTOMER_EXPLANATIONS_PATH = (
    "outputs/reports/customer_churn_explanations.csv"
)

SHAP_VALUES_PATH = (
    "outputs/reports/shap_values.csv"
)

SHAP_SUMMARY_PATH = (
    "outputs/figures/shap_summary.png"
)

SHAP_BAR_PATH = (
    "outputs/figures/shap_bar.png"
)

SHAP_WATERFALL_PATH = (
    "outputs/figures/shap_high_risk_customer_waterfall.png"
)

SHAP_DEPENDENCE_PATH = (
    "outputs/figures/shap_recency_dependence.png"
)

SHAP_REPORT_PATH = (
    "outputs/reports/shap_explainability_report.txt"
)

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

print("=" * 70)
print("SHAP EXPLAINABILITY — CUSTOMER CHURN MODEL")
print("=" * 70)

# ============================================================
# Validate Required Files
# ============================================================

required_files = {
    "Final churn model": MODEL_PATH,
    "Final model feature list": FEATURES_PATH,
    "Churn feature dataset": DATA_PATH,
}

print("\nChecking required files...")

for file_name, file_path in required_files.items():

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"{file_name} was not found:\n"
            f"{file_path}"
        )

    print(
        f"✔ {file_name}: {file_path}"
    )

# ============================================================
# Load Final Model
# ============================================================

print("\nLoading final churn model...")

model = joblib.load(
    MODEL_PATH
)

if not hasattr(
    model,
    "predict_proba",
):
    raise TypeError(
        "The loaded final model does not support "
        "predict_proba()."
    )

print(
    f"Model Type : {type(model).__name__}"
)

# ============================================================
# Load Final Feature Order
# ============================================================

print("\nLoading final model features...")

features = joblib.load(
    FEATURES_PATH
)

if not isinstance(
    features,
    (list, tuple),
):
    raise TypeError(
        "The saved feature configuration must "
        "be a list or tuple."
    )

features = list(features)

if not features:
    raise ValueError(
        "The final model feature list is empty."
    )

if len(features) != len(set(features)):
    raise ValueError(
        "Duplicate feature names were found in "
        "the final model feature list."
    )

print(
    f"Final Feature Count : {len(features)}"
)

for position, feature in enumerate(
    features,
    start=1,
):
    print(
        f"{position:>2}. {feature}"
    )

# ============================================================
# Validate Model Feature Count
# ============================================================

model_feature_count = getattr(
    model,
    "n_features_in_",
    None,
)

if model_feature_count is not None:

    model_feature_count = int(
        model_feature_count
    )

    if model_feature_count != len(features):
        raise ValueError(
            "Saved feature count does not match "
            "the model feature count.\n"
            f"Model expects: {model_feature_count}\n"
            f"Saved features: {len(features)}"
        )

print(
    "\nModel and feature configuration match."
)

# ============================================================
# Load Churn Dataset
# ============================================================

print("\nLoading churn feature dataset...")

df = pd.read_csv(
    DATA_PATH
)

if df.empty:
    raise ValueError(
        "The churn feature dataset is empty."
    )

print(
    f"Dataset Shape : {df.shape}"
)

# ============================================================
# Validate Dataset Columns
# ============================================================

required_columns = (
    [IDENTIFIER, TARGET]
    + features
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Required columns are missing from "
        "the churn dataset:\n"
        f"{missing_columns}"
    )

df[IDENTIFIER] = (
    df[IDENTIFIER]
    .astype(str)
    .str.strip()
)

df[TARGET] = (
    df[TARGET]
    .astype(int)
)

if df[TARGET].nunique() != 2:
    raise ValueError(
        "The churn target must contain "
        "exactly two classes."
    )

duplicate_customers = df[
    IDENTIFIER
].duplicated().sum()

if duplicate_customers > 0:
    raise ValueError(
        f"The dataset contains "
        f"{duplicate_customers} duplicate CustomerID values."
    )

print(
    "Dataset validation completed."
)

# ============================================================
# Prepare Feature Matrix
# ============================================================

X = df[
    features
].copy()

y = df[
    TARGET
].copy()

customer_ids = df[
    IDENTIFIER
].copy()

# Replace infinite values before explanation.
X.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True,
)

missing_value_count = int(
    X.isna().sum().sum()
)

if missing_value_count > 0:

    print(
        f"\nMissing feature values found: "
        f"{missing_value_count}"
    )

    print(
        "Filling missing values with "
        "feature medians..."
    )

    for feature in features:

        if X[feature].isna().any():

            median_value = X[
                feature
            ].median()

            if pd.isna(median_value):
                raise ValueError(
                    f"Feature '{feature}' contains "
                    "only missing values."
                )

            X[feature] = X[
                feature
            ].fillna(
                median_value
            )

# Ensure every feature is numeric.
non_numeric_features = [
    feature
    for feature in features
    if not pd.api.types.is_numeric_dtype(
        X[feature]
    )
]

if non_numeric_features:
    raise TypeError(
        "SHAP input contains non-numeric features:\n"
        f"{non_numeric_features}"
    )

print(
    f"\nCustomers Available : {len(X)}"
)

print(
    f"Churn Customers     : {(y == 1).sum()}"
)

print(
    f"Non-Churn Customers : {(y == 0).sum()}"
)

# ============================================================
# Generate Churn Probabilities
# ============================================================

print("\nGenerating churn probabilities...")

churn_probabilities = model.predict_proba(
    X
)[:, 1]

predicted_churn = (
    churn_probabilities >= 0.50
).astype(int)

prediction_df = pd.DataFrame(
    {
        IDENTIFIER: customer_ids,
        "ActualChurn": y,
        "PredictedChurn": predicted_churn,
        "ChurnProbability": churn_probabilities,
    }
)

prediction_df.sort_values(
    by="ChurnProbability",
    ascending=False,
    inplace=True,
)

prediction_df.reset_index(
    drop=True,
    inplace=True,
)

prediction_df[
    "RiskRank"
] = np.arange(
    1,
    len(prediction_df) + 1,
)

print(
    "Churn probabilities generated successfully."
)

# ============================================================
# Select Representative SHAP Sample
# ============================================================

sample_size = min(
    SHAP_SAMPLE_SIZE,
    len(X),
)

X_shap = X.sample(
    n=sample_size,
    random_state=RANDOM_STATE,
)

sample_indices = X_shap.index

sample_customer_ids = customer_ids.loc[
    sample_indices
].reset_index(
    drop=True
)

sample_actual_churn = y.loc[
    sample_indices
].reset_index(
    drop=True
)

sample_probabilities = churn_probabilities[
    sample_indices
]

X_shap = X_shap.reset_index(
    drop=True
)

print(
    f"\nSHAP Sample Size : "
    f"{len(X_shap)}"
)

# ============================================================
# Create SHAP Explainer
# ============================================================

print("\nCreating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(
    model
)

print(
    "SHAP explainer created successfully."
)

# ============================================================
# Calculate SHAP Values
# ============================================================

print("\nCalculating SHAP values...")

shap_explanation = explainer(
    X_shap
)

# Some model/library combinations may return an additional
# output dimension for binary classification. Keep the churn
# class explanation when that occurs.
if (
    len(shap_explanation.values.shape) == 3
    and shap_explanation.values.shape[2] == 2
):

    shap_explanation = (
        shap_explanation[:, :, 1]
    )

if shap_explanation.values.shape[0] != len(
    X_shap
):
    raise ValueError(
        "SHAP explanation sample count does not "
        "match the input sample count."
    )

if shap_explanation.values.shape[1] != len(
    features
):
    raise ValueError(
        "SHAP explanation feature count does not "
        "match the final model feature count."
    )

print(
    "SHAP values calculated successfully."
)

print(
    f"SHAP Values Shape : "
    f"{shap_explanation.values.shape}"
)

print(
    "\nProceeding to global feature importance "
    "and SHAP visualizations..."
)

# ============================================================
# Calculate Global SHAP Feature Importance
# ============================================================

mean_absolute_shap = np.abs(
    shap_explanation.values
).mean(axis=0)

mean_signed_shap = (
    shap_explanation.values.mean(
        axis=0
    )
)

total_importance = float(
    mean_absolute_shap.sum()
)

if total_importance > 0:
    normalized_importance = (
        mean_absolute_shap
        / total_importance
    )
else:
    normalized_importance = np.zeros(
        len(features)
    )

shap_importance_df = pd.DataFrame(
    {
        "Feature": features,
        "MeanAbsoluteSHAP": (
            mean_absolute_shap
        ),
        "MeanSignedSHAP": (
            mean_signed_shap
        ),
        "NormalizedImportance": (
            normalized_importance
        ),
        "ImportancePercentage": (
            normalized_importance * 100
        ),
    }
)

shap_importance_df.sort_values(
    by="MeanAbsoluteSHAP",
    ascending=False,
    inplace=True,
)

shap_importance_df.reset_index(
    drop=True,
    inplace=True,
)

shap_importance_df[
    "Rank"
] = np.arange(
    1,
    len(shap_importance_df) + 1,
)

shap_importance_df[
    "CumulativeImportance"
] = shap_importance_df[
    "NormalizedImportance"
].cumsum()

# Reorder report columns.
shap_importance_df = (
    shap_importance_df[
        [
            "Rank",
            "Feature",
            "MeanAbsoluteSHAP",
            "MeanSignedSHAP",
            "NormalizedImportance",
            "ImportancePercentage",
            "CumulativeImportance",
        ]
    ]
)

# ============================================================
# Save Global SHAP Feature Importance
# ============================================================

shap_importance_df.to_csv(
    SHAP_IMPORTANCE_PATH,
    index=False,
)

print(
    f"\nSHAP feature importance saved: "
    f"{SHAP_IMPORTANCE_PATH}"
)

# ============================================================
# Display Global Feature Ranking
# ============================================================

print("\n" + "=" * 70)
print("GLOBAL SHAP FEATURE IMPORTANCE")
print("=" * 70)

for _, row in (
    shap_importance_df.iterrows()
):

    print(
        f"{int(row['Rank']):>2}. "
        f"{row['Feature']:<24} "
        f"Mean |SHAP|: "
        f"{row['MeanAbsoluteSHAP']:.6f} | "
        f"Importance: "
        f"{row['ImportancePercentage']:.2f}%"
    )

most_important_feature = (
    shap_importance_df.iloc[0][
        "Feature"
    ]
)

print(
    f"\nMost Important Feature : "
    f"{most_important_feature}"
)

# ============================================================
# Create SHAP Values DataFrame
# ============================================================

shap_values_df = pd.DataFrame(
    shap_explanation.values,
    columns=[
        f"SHAP_{feature}"
        for feature in features
    ],
)

feature_values_df = X_shap.copy()

feature_values_df.columns = [
    f"Value_{feature}"
    for feature in features
]

# Normalize base values to one value per customer.
base_values = np.asarray(
    shap_explanation.base_values
)

if base_values.ndim == 0:

    base_values = np.repeat(
        float(base_values),
        len(X_shap),
    )

elif base_values.ndim == 1:

    if len(base_values) == 1:
        base_values = np.repeat(
            float(base_values[0]),
            len(X_shap),
        )

    elif len(base_values) != len(
        X_shap
    ):
        raise ValueError(
            "Unexpected SHAP base-value shape."
        )

elif base_values.ndim == 2:

    # Binary classification may return one base value
    # per class. Use the churn-class base value.
    if base_values.shape[1] == 2:
        base_values = base_values[
            :,
            1,
        ]

    elif base_values.shape[1] == 1:
        base_values = base_values[
            :,
            0,
        ]

    else:
        raise ValueError(
            "Unexpected multi-output SHAP "
            "base-value shape."
        )

else:
    raise ValueError(
        "Unsupported SHAP base-value dimensions."
    )

sample_results_df = pd.DataFrame(
    {
        IDENTIFIER: (
            sample_customer_ids
        ),
        "ActualChurn": (
            sample_actual_churn
        ),
        "ChurnProbability": (
            sample_probabilities
        ),
        "SHAPBaseValue": (
            base_values
        ),
    }
)

sample_results_df = pd.concat(
    [
        sample_results_df,
        feature_values_df,
        shap_values_df,
    ],
    axis=1,
)

sample_results_df.sort_values(
    by="ChurnProbability",
    ascending=False,
    inplace=True,
)

sample_results_df.reset_index(
    drop=True,
    inplace=True,
)

sample_results_df.to_csv(
    SHAP_VALUES_PATH,
    index=False,
)

print(
    f"Detailed SHAP values saved: "
    f"{SHAP_VALUES_PATH}"
)

# ============================================================
# Generate SHAP Beeswarm Summary Plot
# ============================================================

print(
    "\nGenerating SHAP beeswarm "
    "summary plot..."
)

plt.close("all")

shap.plots.beeswarm(
    shap_explanation,
    max_display=len(features),
    show=False,
    plot_size=(
        11,
        max(
            6,
            len(features) * 0.8,
        ),
    ),
)

plt.title(
    "SHAP Summary — Customer Churn Model",
    pad=20,
)

plt.tight_layout()

plt.savefig(
    SHAP_SUMMARY_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    f"SHAP summary plot saved: "
    f"{SHAP_SUMMARY_PATH}"
)

# ============================================================
# Generate Global SHAP Bar Plot
# ============================================================

print(
    "Generating global SHAP "
    "importance bar plot..."
)

plt.close("all")

shap.plots.bar(
    shap_explanation,
    max_display=len(features),
    show=False,
)

current_figure = plt.gcf()

current_figure.set_size_inches(
    10,
    max(
        6,
        len(features) * 0.8,
    ),
)

plt.title(
    "Global SHAP Feature Importance",
    pad=20,
)

plt.tight_layout()

plt.savefig(
    SHAP_BAR_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    f"SHAP bar plot saved: "
    f"{SHAP_BAR_PATH}"
)

# ============================================================
# Select Dependence Feature
# ============================================================

# Recency is the preferred business feature.
# If it is unavailable, use the globally most
# important feature identified by SHAP.

if "Recency" in features:
    dependence_feature = "Recency"
else:
    dependence_feature = (
        most_important_feature
    )

print(
    f"\nDependence Feature : "
    f"{dependence_feature}"
)

# ============================================================
# Generate SHAP Dependence Scatter Plot
# ============================================================

print(
    "Generating SHAP dependence plot..."
)

plt.close("all")

# Passing the complete Explanation object to `color`
# lets SHAP automatically identify a likely interacting
# feature for colouring the points.

shap.plots.scatter(
    shap_explanation[
        :,
        dependence_feature,
    ],
    color=shap_explanation,
    show=False,
)

current_figure = plt.gcf()

current_figure.set_size_inches(
    10,
    7,
)

plt.title(
    f"SHAP Dependence — "
    f"{dependence_feature}",
    pad=20,
)

plt.tight_layout()

plt.savefig(
    SHAP_DEPENDENCE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    f"SHAP dependence plot saved: "
    f"{SHAP_DEPENDENCE_PATH}"
)

# ============================================================
# Calculate Positive and Negative Contributions
# ============================================================

positive_mean_contribution = np.where(
    shap_explanation.values > 0,
    shap_explanation.values,
    0,
).mean(axis=0)

negative_mean_contribution = np.where(
    shap_explanation.values < 0,
    shap_explanation.values,
    0,
).mean(axis=0)

direction_df = pd.DataFrame(
    {
        "Feature": features,
        "MeanPositiveContribution": (
            positive_mean_contribution
        ),
        "MeanNegativeContribution": (
            negative_mean_contribution
        ),
    }
)

shap_importance_df = (
    shap_importance_df.merge(
        direction_df,
        on="Feature",
        how="left",
        validate="one_to_one",
    )
)

# Save again with the directional contribution columns.
shap_importance_df.to_csv(
    SHAP_IMPORTANCE_PATH,
    index=False,
)

# ============================================================
# Global Explanation Summary
# ============================================================

top_three_features = (
    shap_importance_df.head(3)[
        "Feature"
    ].tolist()
)

print("\n" + "=" * 70)
print("GLOBAL EXPLANATION SUMMARY")
print("=" * 70)

print(
    f"Customers Explained     : "
    f"{len(X_shap)}"
)

print(
    f"Features Explained      : "
    f"{len(features)}"
)

print(
    f"Most Important Feature  : "
    f"{most_important_feature}"
)

print(
    "Top Three Features      : "
    + ", ".join(
        top_three_features
    )
)

print(
    f"Dependence Plot Feature : "
    f"{dependence_feature}"
)

print("\nGenerated Global Outputs")
print("-" * 70)

global_outputs = [
    SHAP_IMPORTANCE_PATH,
    SHAP_VALUES_PATH,
    SHAP_SUMMARY_PATH,
    SHAP_BAR_PATH,
    SHAP_DEPENDENCE_PATH,
]

for output_path in global_outputs:
    print(
        f"✔ {output_path}"
    )

print(
    "\nGlobal SHAP analysis completed "
    "successfully."
)

print(
    "Proceeding to individual high-risk "
    "customer explanations..."
)
# ============================================================
# Normalize Binary SHAP Explanation
# ============================================================

def normalize_binary_explanation(
    explanation,
):
    """
    Convert a binary multi-output SHAP explanation into
    the churn-class explanation when necessary.
    """

    explanation_values = np.asarray(
        explanation.values
    )

    if explanation_values.ndim == 3:

        if explanation_values.shape[2] == 2:
            return explanation[
                :,
                :,
                1,
            ]

        raise ValueError(
            "Unexpected SHAP output-class dimension: "
            f"{explanation_values.shape}"
        )

    if explanation_values.ndim != 2:
        raise ValueError(
            "Expected a two-dimensional SHAP explanation, "
            f"but received shape {explanation_values.shape}."
        )

    return explanation


# ============================================================
# Normalize SHAP Base Values
# ============================================================

def normalize_base_values(
    explanation,
    customer_count,
):
    """
    Return one SHAP base value for each customer.
    """

    values = np.asarray(
        explanation.base_values
    )

    if values.ndim == 0:

        return np.repeat(
            float(values),
            customer_count,
        )

    if values.ndim == 1:

        if len(values) == customer_count:
            return values.astype(float)

        if len(values) == 1:
            return np.repeat(
                float(values[0]),
                customer_count,
            )

    if values.ndim == 2:

        if values.shape == (
            customer_count,
            2,
        ):
            return values[
                :,
                1,
            ].astype(float)

        if values.shape == (
            customer_count,
            1,
        ):
            return values[
                :,
                0,
            ].astype(float)

        if values.shape == (1, 2):
            return np.repeat(
                float(values[0, 1]),
                customer_count,
            )

        if values.shape == (1, 1):
            return np.repeat(
                float(values[0, 0]),
                customer_count,
            )

    raise ValueError(
        "Unable to normalize SHAP base values. "
        f"Received shape: {values.shape}"
    )


# ============================================================
# Select Highest-Risk Customers
# ============================================================

top_customer_count = min(
    TOP_CUSTOMER_COUNT,
    len(X),
)

top_customer_positions = np.argsort(
    churn_probabilities
)[::-1][
    :top_customer_count
]

X_top_risk = X.iloc[
    top_customer_positions
].copy()

top_customer_ids = customer_ids.iloc[
    top_customer_positions
].reset_index(
    drop=True
)

top_actual_churn = y.iloc[
    top_customer_positions
].reset_index(
    drop=True
)

top_predictions = predicted_churn[
    top_customer_positions
]

top_probabilities = churn_probabilities[
    top_customer_positions
]

X_top_risk.reset_index(
    drop=True,
    inplace=True,
)

print("\n" + "=" * 70)
print("HIGH-RISK CUSTOMER EXPLANATIONS")
print("=" * 70)

print(
    f"Customers Selected : "
    f"{top_customer_count}"
)

print(
    f"Probability Range  : "
    f"{top_probabilities.min():.4f} "
    f"to {top_probabilities.max():.4f}"
)

# ============================================================
# Calculate SHAP Values for High-Risk Customers
# ============================================================

print(
    "\nCalculating SHAP values for "
    "high-risk customers..."
)

top_shap_explanation = explainer(
    X_top_risk
)

top_shap_explanation = (
    normalize_binary_explanation(
        top_shap_explanation
    )
)

top_shap_values = np.asarray(
    top_shap_explanation.values
)

if top_shap_values.shape != (
    top_customer_count,
    len(features),
):
    raise ValueError(
        "Unexpected high-risk SHAP value shape: "
        f"{top_shap_values.shape}"
    )

top_base_values = normalize_base_values(
    top_shap_explanation,
    top_customer_count,
)

# TreeExplainer uses raw model output by default.
top_raw_model_outputs = (
    top_base_values
    + top_shap_values.sum(
        axis=1
    )
)

print(
    "High-risk SHAP values calculated successfully."
)

# ============================================================
# Local Driver Extraction Function
# ============================================================

def extract_local_drivers(
    shap_row,
    feature_row,
    positive=True,
    maximum_drivers=3,
):
    """
    Extract the strongest positive or negative SHAP drivers.

    Positive SHAP values increase the churn model score.
    Negative SHAP values reduce the churn model score.
    """

    shap_row = np.asarray(
        shap_row,
        dtype=float,
    )

    if positive:
        eligible_indices = np.where(
            shap_row > 0
        )[0]
    else:
        eligible_indices = np.where(
            shap_row < 0
        )[0]

    if len(eligible_indices) == 0:
        return []

    sorted_indices = eligible_indices[
        np.argsort(
            np.abs(
                shap_row[
                    eligible_indices
                ]
            )
        )[::-1]
    ]

    selected_indices = sorted_indices[
        :maximum_drivers
    ]

    driver_records = []

    for feature_index in selected_indices:

        feature_name = features[
            feature_index
        ]

        feature_value = float(
            feature_row.iloc[
                feature_index
            ]
        )

        shap_value = float(
            shap_row[
                feature_index
            ]
        )

        driver_records.append(
            {
                "Feature": feature_name,
                "FeatureValue": feature_value,
                "SHAPValue": shap_value,
            }
        )

    return driver_records


# ============================================================
# Driver Formatting Function
# ============================================================

def format_driver_records(
    driver_records,
):
    """
    Convert driver information into a readable string.
    """

    if not driver_records:
        return "None"

    formatted_drivers = []

    for driver in driver_records:

        formatted_drivers.append(
            f"{driver['Feature']}="
            f"{driver['FeatureValue']:.4f} "
            f"(SHAP {driver['SHAPValue']:+.4f})"
        )

    return " | ".join(
        formatted_drivers
    )


# ============================================================
# Build Customer-Level Explanation Records
# ============================================================

customer_explanation_records = []

for customer_position in range(
    top_customer_count
):

    customer_shap_values = (
        top_shap_values[
            customer_position
        ]
    )

    customer_feature_values = (
        X_top_risk.iloc[
            customer_position
        ]
    )

    positive_drivers = (
        extract_local_drivers(
            shap_row=customer_shap_values,
            feature_row=customer_feature_values,
            positive=True,
            maximum_drivers=3,
        )
    )

    negative_drivers = (
        extract_local_drivers(
            shap_row=customer_shap_values,
            feature_row=customer_feature_values,
            positive=False,
            maximum_drivers=3,
        )
    )

    dominant_feature_index = int(
        np.argmax(
            np.abs(
                customer_shap_values
            )
        )
    )

    dominant_feature = features[
        dominant_feature_index
    ]

    dominant_feature_value = float(
        customer_feature_values.iloc[
            dominant_feature_index
        ]
    )

    dominant_shap_value = float(
        customer_shap_values[
            dominant_feature_index
        ]
    )

    positive_contribution_sum = float(
        np.clip(
            customer_shap_values,
            0,
            None,
        ).sum()
    )

    negative_contribution_sum = float(
        np.clip(
            customer_shap_values,
            None,
            0,
        ).sum()
    )

    positive_driver_text = (
        format_driver_records(
            positive_drivers
        )
    )

    negative_driver_text = (
        format_driver_records(
            negative_drivers
        )
    )

    explanation_summary = (
        f"Dominant feature: {dominant_feature}="
        f"{dominant_feature_value:.4f} "
        f"with SHAP contribution "
        f"{dominant_shap_value:+.4f}. "
        f"Strongest risk-increasing drivers: "
        f"{positive_driver_text}. "
        f"Strongest risk-reducing drivers: "
        f"{negative_driver_text}."
    )

    customer_record = {
        "RiskRank": (
            customer_position + 1
        ),
        IDENTIFIER: (
            top_customer_ids.iloc[
                customer_position
            ]
        ),
        "ActualChurn": int(
            top_actual_churn.iloc[
                customer_position
            ]
        ),
        "PredictedChurn": int(
            top_predictions[
                customer_position
            ]
        ),
        "ChurnProbability": float(
            top_probabilities[
                customer_position
            ]
        ),
        "SHAPBaseValueRaw": float(
            top_base_values[
                customer_position
            ]
        ),
        "SHAPModelOutputRaw": float(
            top_raw_model_outputs[
                customer_position
            ]
        ),
        "PositiveContributionSum": (
            positive_contribution_sum
        ),
        "NegativeContributionSum": (
            negative_contribution_sum
        ),
        "DominantFeature": (
            dominant_feature
        ),
        "DominantFeatureValue": (
            dominant_feature_value
        ),
        "DominantSHAPValue": (
            dominant_shap_value
        ),
        "RiskIncreasingDrivers": (
            positive_driver_text
        ),
        "RiskReducingDrivers": (
            negative_driver_text
        ),
        "ExplanationSummary": (
            explanation_summary
        ),
    }

    # Include each original feature and local SHAP value.
    for feature_index, feature in enumerate(
        features
    ):

        customer_record[
            f"Value_{feature}"
        ] = float(
            customer_feature_values[
                feature
            ]
        )

        customer_record[
            f"SHAP_{feature}"
        ] = float(
            customer_shap_values[
                feature_index
            ]
        )

    customer_explanation_records.append(
        customer_record
    )

customer_explanations_df = pd.DataFrame(
    customer_explanation_records
)

customer_explanations_df.sort_values(
    by="RiskRank",
    inplace=True,
)

customer_explanations_df.reset_index(
    drop=True,
    inplace=True,
)

# ============================================================
# Save Customer Explanations
# ============================================================

customer_explanations_df.to_csv(
    CUSTOMER_EXPLANATIONS_PATH,
    index=False,
)

print(
    f"\nCustomer explanations saved: "
    f"{CUSTOMER_EXPLANATIONS_PATH}"
)

# ============================================================
# Display Top Customer Explanation
# ============================================================

highest_risk_customer = (
    customer_explanations_df.iloc[0]
)

print("\n" + "=" * 70)
print("HIGHEST-RISK CUSTOMER")
print("=" * 70)

print(
    f"CustomerID             : "
    f"{highest_risk_customer[IDENTIFIER]}"
)

print(
    f"Risk Rank              : "
    f"{int(highest_risk_customer['RiskRank'])}"
)

print(
    f"Actual Churn           : "
    f"{int(highest_risk_customer['ActualChurn'])}"
)

print(
    f"Predicted Churn        : "
    f"{int(highest_risk_customer['PredictedChurn'])}"
)

print(
    f"Churn Probability      : "
    f"{highest_risk_customer['ChurnProbability']:.4f}"
)

print(
    f"Dominant Feature       : "
    f"{highest_risk_customer['DominantFeature']}"
)

print(
    f"Dominant SHAP Value    : "
    f"{highest_risk_customer['DominantSHAPValue']:+.4f}"
)

print(
    "\nRisk-Increasing Drivers"
)

print(
    highest_risk_customer[
        "RiskIncreasingDrivers"
    ]
)

print(
    "\nRisk-Reducing Drivers"
)

print(
    highest_risk_customer[
        "RiskReducingDrivers"
    ]
)

# ============================================================
# Generate Highest-Risk Customer Waterfall Plot
# ============================================================

print(
    "\nGenerating highest-risk customer "
    "waterfall plot..."
)

plt.close("all")

shap.plots.waterfall(
    top_shap_explanation[0],
    max_display=len(features),
    show=False,
)

current_figure = plt.gcf()

current_figure.set_size_inches(
    11,
    max(
        7,
        len(features) * 1.0,
    ),
)

plt.title(
    "SHAP Waterfall — Highest-Risk Customer\n"
    f"CustomerID: "
    f"{highest_risk_customer[IDENTIFIER]} | "
    f"Probability: "
    f"{highest_risk_customer['ChurnProbability']:.4f}",
    pad=20,
)

plt.tight_layout()

plt.savefig(
    SHAP_WATERFALL_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    f"SHAP waterfall plot saved: "
    f"{SHAP_WATERFALL_PATH}"
)

# ============================================================
# Calculate Explanation Statistics
# ============================================================

actual_churners_in_top_group = int(
    customer_explanations_df[
        "ActualChurn"
    ].sum()
)

predicted_churners_in_top_group = int(
    customer_explanations_df[
        "PredictedChurn"
    ].sum()
)

top_group_actual_churn_rate = float(
    customer_explanations_df[
        "ActualChurn"
    ].mean()
)

top_group_average_probability = float(
    customer_explanations_df[
        "ChurnProbability"
    ].mean()
)

dominant_feature_counts = (
    customer_explanations_df[
        "DominantFeature"
    ]
    .value_counts()
)

# ============================================================
# Write SHAP Explainability Report
# ============================================================

with open(
    SHAP_REPORT_PATH,
    "w",
    encoding="utf-8",
) as report:

    report.write("RetailPulse\n")

    report.write(
        "F04 Customer Churn Prediction\n"
    )

    report.write(
        "SHAP Model Explainability Report\n"
    )

    report.write("=" * 75)
    report.write("\n\n")

    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

    report.write("Model Information\n")
    report.write("-" * 75)
    report.write("\n")

    report.write(
        f"Model Path: {MODEL_PATH}\n"
    )

    report.write(
        f"Model Type: "
        f"{type(model).__name__}\n"
    )

    report.write(
        f"Dataset Path: {DATA_PATH}\n"
    )

    report.write(
        f"Dataset Shape: {df.shape}\n"
    )

    report.write(
        f"Customer Count: {len(X)}\n"
    )

    report.write(
        f"Feature Count: {len(features)}\n"
    )

    report.write(
        f"SHAP Sample Size: "
        f"{len(X_shap)}\n"
    )

    report.write(
        f"Individually Explained Customers: "
        f"{top_customer_count}\n\n"
    )

    # --------------------------------------------------------
    # SHAP interpretation note
    # --------------------------------------------------------

    report.write(
        "SHAP Interpretation\n"
    )

    report.write("-" * 75)
    report.write("\n")

    report.write(
        "Positive SHAP values increase the model's "
        "raw churn score.\n"
    )

    report.write(
        "Negative SHAP values reduce the model's "
        "raw churn score.\n"
    )

    report.write(
        "TreeExplainer was created with its default "
        "raw model-output configuration. Therefore, "
        "the SHAP values are additive in raw model-output "
        "space rather than probability space.\n"
    )

    report.write(
        "SHAP explanations describe model behaviour "
        "and should not be interpreted as proof of "
        "causal relationships.\n\n"
    )

    # --------------------------------------------------------
    # Model features
    # --------------------------------------------------------

    report.write("Model Features\n")
    report.write("-" * 75)
    report.write("\n")

    for position, feature in enumerate(
        features,
        start=1,
    ):

        report.write(
            f"{position:>2}. {feature}\n"
        )

    report.write("\n")

    # --------------------------------------------------------
    # Global SHAP importance
    # --------------------------------------------------------

    report.write(
        "Global SHAP Feature Importance\n"
    )

    report.write("-" * 75)
    report.write("\n")

    report.write(
        shap_importance_df.to_string(
            index=False,
        )
    )

    report.write("\n\n")

    report.write(
        f"Most Important Feature: "
        f"{most_important_feature}\n"
    )

    report.write(
        "Top Three Features: "
        + ", ".join(
            top_three_features
        )
        + "\n"
    )

    report.write(
        f"Dependence Feature: "
        f"{dependence_feature}\n\n"
    )

    # --------------------------------------------------------
    # High-risk customer summary
    # --------------------------------------------------------

    report.write(
        "High-Risk Customer Summary\n"
    )

    report.write("-" * 75)
    report.write("\n")

    report.write(
        f"Customers Evaluated: "
        f"{top_customer_count}\n"
    )

    report.write(
        f"Actual Churners: "
        f"{actual_churners_in_top_group}\n"
    )

    report.write(
        f"Predicted Churners: "
        f"{predicted_churners_in_top_group}\n"
    )

    report.write(
        f"Actual Churn Rate: "
        f"{top_group_actual_churn_rate:.4f}\n"
    )

    report.write(
        f"Average Predicted Probability: "
        f"{top_group_average_probability:.4f}\n\n"
    )

    # --------------------------------------------------------
    # Dominant feature counts
    # --------------------------------------------------------

    report.write(
        "Dominant Features Among High-Risk Customers\n"
    )

    report.write("-" * 75)
    report.write("\n")

    for feature, count in (
        dominant_feature_counts.items()
    ):

        report.write(
            f"{feature}: {count}\n"
        )

    report.write("\n")

    # --------------------------------------------------------
    # Individual explanations
    # --------------------------------------------------------

    report.write(
        "Individual High-Risk Customer Explanations\n"
    )

    report.write("-" * 75)
    report.write("\n")

    report_columns = [
        "RiskRank",
        IDENTIFIER,
        "ActualChurn",
        "PredictedChurn",
        "ChurnProbability",
        "DominantFeature",
        "DominantFeatureValue",
        "DominantSHAPValue",
        "RiskIncreasingDrivers",
        "RiskReducingDrivers",
    ]

    report.write(
        customer_explanations_df[
            report_columns
        ].to_string(
            index=False,
        )
    )

    report.write("\n\n")

    # --------------------------------------------------------
    # Generated files
    # --------------------------------------------------------

    report.write("Generated Outputs\n")
    report.write("-" * 75)
    report.write("\n")

    report_output_paths = [
        SHAP_IMPORTANCE_PATH,
        CUSTOMER_EXPLANATIONS_PATH,
        SHAP_VALUES_PATH,
        SHAP_SUMMARY_PATH,
        SHAP_BAR_PATH,
        SHAP_DEPENDENCE_PATH,
        SHAP_WATERFALL_PATH,
    ]

    for output_path in report_output_paths:

        report.write(
            f"- {output_path}\n"
        )

print(
    f"SHAP explainability report saved: "
    f"{SHAP_REPORT_PATH}"
)

# ============================================================
# Final Console Summary
# ============================================================

print("\n" + "=" * 70)
print("SHAP EXPLAINABILITY SUMMARY")
print("=" * 70)

print(
    f"Model Type                 : "
    f"{type(model).__name__}"
)

print(
    f"Total Customers            : "
    f"{len(X)}"
)

print(
    f"Global SHAP Sample         : "
    f"{len(X_shap)}"
)

print(
    f"Explained High-Risk Users  : "
    f"{top_customer_count}"
)

print(
    f"Most Important Feature     : "
    f"{most_important_feature}"
)

print(
    f"Top Three Features         : "
    f"{', '.join(top_three_features)}"
)

print(
    f"Actual High-Risk Churners  : "
    f"{actual_churners_in_top_group}"
)

print(
    f"Top-Group Churn Rate       : "
    f"{top_group_actual_churn_rate:.4f}"
)

print(
    f"Average Churn Probability  : "
    f"{top_group_average_probability:.4f}"
)

print("\nDominant Feature Counts")
print("-" * 70)

for feature, count in (
    dominant_feature_counts.items()
):

    print(
        f"{feature:<25}: {count}"
    )

# ============================================================
# Generated Files
# ============================================================

generated_files = [
    SHAP_IMPORTANCE_PATH,
    CUSTOMER_EXPLANATIONS_PATH,
    SHAP_VALUES_PATH,
    SHAP_REPORT_PATH,
    SHAP_SUMMARY_PATH,
    SHAP_BAR_PATH,
    SHAP_DEPENDENCE_PATH,
    SHAP_WATERFALL_PATH,
]

print("\nGenerated Files")
print("-" * 70)

for generated_file in generated_files:

    print(
        f"✔ {generated_file}"
    )

# ============================================================
# Completion
# ============================================================

print("\n" + "=" * 70)

print(
    "F04 SHAP MODEL EXPLAINABILITY "
    "COMPLETED SUCCESSFULLY"
)

print("=" * 70)