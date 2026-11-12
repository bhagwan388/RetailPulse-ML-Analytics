# ============================================================
# RetailPulse
# F04 Customer Churn Prediction
# Target Leakage Audit
# ============================================================

from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "churn_features.csv"
)

ADVANCED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "advanced_churn_features.csv"
)

TARGET = "Churn"
IDENTIFIER = "CustomerID"

REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

FIGURE_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
)

SOURCE_DIRECTORY = (
    PROJECT_ROOT
    / "src"
    / "churn"
)

NOTEBOOK_DIRECTORY = (
    PROJECT_ROOT
    / "notebooks"
)

REPORT_PATH = (
    REPORT_DIRECTORY
    / "target_leakage_audit.txt"
)

FEATURE_AUDIT_PATH = (
    REPORT_DIRECTORY
    / "target_leakage_feature_metrics.csv"
)

THRESHOLD_AUDIT_PATH = (
    REPORT_DIRECTORY
    / "recency_threshold_audit.csv"
)

SOURCE_MATCHES_PATH = (
    REPORT_DIRECTORY
    / "churn_definition_source_matches.csv"
)

RANDOM_STATE = 42

REPORT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

print("=" * 70)
print("F04 TARGET LEAKAGE AUDIT")
print("=" * 70)

# ============================================================
# Load Dataset
# ============================================================

print("\nLoading churn dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(
    DATA_PATH
)

if df.empty:
    raise ValueError(
        "The churn dataset is empty."
    )

print(
    f"Dataset Path  : {DATA_PATH}"
)

print(
    f"Dataset Shape : {df.shape}"
)

# ============================================================
# Validate Required Columns
# ============================================================

required_columns = [
    IDENTIFIER,
    TARGET,
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Required columns are missing:\n"
        f"{missing_columns}"
    )

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce",
)

if df[TARGET].isna().any():
    raise ValueError(
        "The churn target contains "
        "non-numeric or missing values."
    )

df[TARGET] = df[TARGET].astype(int)

target_values = sorted(
    df[TARGET].unique().tolist()
)

if target_values != [0, 1]:
    raise ValueError(
        "The churn target must contain "
        f"only 0 and 1. Found: {target_values}"
    )

duplicate_customer_count = int(
    df[IDENTIFIER].duplicated().sum()
)

if duplicate_customer_count > 0:
    print(
        f"\nWarning: {duplicate_customer_count} "
        "duplicate CustomerID values found."
    )

# ============================================================
# Target Distribution
# ============================================================

print("\nTarget Distribution")
print("-" * 70)

target_distribution = (
    df[TARGET]
    .value_counts()
    .sort_index()
)

for target_value, count in (
    target_distribution.items()
):

    percentage = (
        count
        / len(df)
        * 100
    )

    label = (
        "Non-Churn"
        if target_value == 0
        else "Churn"
    )

    print(
        f"{label:<12}: "
        f"{count} "
        f"({percentage:.2f}%)"
    )

# ============================================================
# Prepare Numeric Features
# ============================================================

excluded_columns = [
    IDENTIFIER,
    TARGET,
]

candidate_features = [
    column
    for column in df.columns
    if column not in excluded_columns
]

numeric_features = [
    column
    for column in candidate_features
    if pd.api.types.is_numeric_dtype(
        df[column]
    )
]

non_numeric_features = [
    column
    for column in candidate_features
    if column not in numeric_features
]

print(
    f"\nCandidate Features : "
    f"{len(candidate_features)}"
)

print(
    f"Numeric Features   : "
    f"{len(numeric_features)}"
)

if non_numeric_features:

    print(
        "\nNon-numeric features excluded:"
    )

    for feature in non_numeric_features:
        print(f"- {feature}")

if not numeric_features:
    raise ValueError(
        "No numeric features were found."
    )

# ============================================================
# Clean Numeric Feature Data
# ============================================================

numeric_df = df[
    numeric_features
].copy()

numeric_df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True,
)

for feature in numeric_features:

    if numeric_df[feature].isna().any():

        median_value = numeric_df[
            feature
        ].median()

        if pd.isna(median_value):
            raise ValueError(
                f"Feature '{feature}' "
                "contains only missing values."
            )

        numeric_df[feature] = (
            numeric_df[feature]
            .fillna(median_value)
        )

print(
    "\nNumeric feature preparation completed."
)

# ============================================================
# Source-Code Scan Configuration
# ============================================================

# These patterns search for possible target creation,
# direct use of Recency, threshold rules, and future-window logic.

SOURCE_PATTERNS = {
    "Direct churn assignment": re.compile(
        r"(?:\[|\.)(?:['\"])?Churn(?:['\"])?\]?\s*=",
        re.IGNORECASE,
    ),
    "Recency threshold comparison": re.compile(
        r"Recency.{0,80}(?:>|>=|<|<=).{0,20}\d+",
        re.IGNORECASE,
    ),
    "Threshold before Churn": re.compile(
        r"(?:>|>=|<|<=).{0,30}\d+.{0,100}Churn",
        re.IGNORECASE,
    ),
    "No-purchase logic": re.compile(
        r"(?:no purchase|no_purchase|inactive|inactivity)",
        re.IGNORECASE,
    ),
    "Future observation window": re.compile(
        r"(?:future|observation window|target window|cutoff date)",
        re.IGNORECASE,
    ),
    "Shift or forward target": re.compile(
        r"(?:shift\s*\(|lead\s*\(|future_purchase)",
        re.IGNORECASE,
    ),
}

# ============================================================
# Read Text From Source File
# ============================================================

def read_source_text(
    file_path: Path,
) -> str:
    """
    Read Python, text, or notebook content safely.
    """

    try:

        if file_path.suffix.lower() == ".ipynb":

            notebook_data = json.loads(
                file_path.read_text(
                    encoding="utf-8"
                )
            )

            cell_text = []

            for cell in notebook_data.get(
                "cells",
                [],
            ):

                source = cell.get(
                    "source",
                    [],
                )

                if isinstance(source, list):
                    cell_text.extend(source)
                else:
                    cell_text.append(
                        str(source)
                    )

            return "\n".join(
                cell_text
            )

        return file_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    except Exception as error:

        print(
            f"Could not read "
            f"{file_path}: {error}"
        )

        return ""


# ============================================================
# Scan Project Files for Churn Definition
# ============================================================

print("\nScanning source files for churn-label logic...")

source_files = []

if SOURCE_DIRECTORY.exists():

    source_files.extend(
        SOURCE_DIRECTORY.rglob("*.py")
    )

if NOTEBOOK_DIRECTORY.exists():

    source_files.extend(
        NOTEBOOK_DIRECTORY.rglob("*.ipynb")
    )

source_match_records = []

for file_path in sorted(
    set(source_files)
):

    file_text = read_source_text(
        file_path
    )

    if not file_text:
        continue

    lines = file_text.splitlines()

    for line_number, line in enumerate(
        lines,
        start=1,
    ):

        for pattern_name, pattern in (
            SOURCE_PATTERNS.items()
        ):

            if pattern.search(line):

                relative_path = (
                    file_path.relative_to(
                        PROJECT_ROOT
                    )
                )

                source_match_records.append(
                    {
                        "File": str(
                            relative_path
                        ),
                        "LineNumber": (
                            line_number
                        ),
                        "Pattern": (
                            pattern_name
                        ),
                        "SourceLine": (
                            line.strip()
                        ),
                    }
                )

source_matches_df = pd.DataFrame(
    source_match_records
)

if source_matches_df.empty:

    print(
        "No possible churn-definition "
        "lines were automatically detected."
    )

else:

    source_matches_df.to_csv(
        SOURCE_MATCHES_PATH,
        index=False,
    )

    print(
        f"Possible source matches found: "
        f"{len(source_matches_df)}"
    )

    print(
        f"Saved source matches: "
        f"{SOURCE_MATCHES_PATH}"
    )

    print(
        "\nFirst detected matches"
    )

    print("-" * 70)

    for _, row in (
        source_matches_df
        .head(20)
        .iterrows()
    ):

        print(
            f"{row['File']}:"
            f"{row['LineNumber']} | "
            f"{row['Pattern']} | "
            f"{row['SourceLine']}"
        )

# ============================================================
# Basic Feature Correlation With Target
# ============================================================

correlation_records = []

for feature in numeric_features:

    feature_values = numeric_df[
        feature
    ]

    correlation = feature_values.corr(
        df[TARGET]
    )

    correlation_records.append(
        {
            "Feature": feature,
            "PearsonCorrelation": (
                correlation
            ),
            "AbsoluteCorrelation": (
                abs(correlation)
                if not pd.isna(correlation)
                else np.nan
            ),
            "UniqueValues": int(
                feature_values.nunique()
            ),
        }
    )

correlation_df = pd.DataFrame(
    correlation_records
)

correlation_df.sort_values(
    by="AbsoluteCorrelation",
    ascending=False,
    inplace=True,
)

correlation_df.reset_index(
    drop=True,
    inplace=True,
)

print("\n" + "=" * 70)
print("FEATURE–TARGET CORRELATION")
print("=" * 70)

for _, row in (
    correlation_df.head(10).iterrows()
):

    print(
        f"{row['Feature']:<24} "
        f"Correlation: "
        f"{row['PearsonCorrelation']:+.4f}"
    )

print(
    "\nProceeding to univariate AUC, "
    "threshold reconstruction, and leakage risk scoring..."
)

# ============================================================
# Additional Imports for Leakage Metrics
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
)

# ============================================================
# Binary Classification Metric Function
# ============================================================

def calculate_binary_metrics(
    y_true,
    predictions,
):
    """
    Calculate classification metrics for a reconstructed
    churn target.
    """

    y_true_array = np.asarray(
        y_true,
        dtype=int,
    )

    prediction_array = np.asarray(
        predictions,
        dtype=int,
    )

    if len(y_true_array) != len(
        prediction_array
    ):
        raise ValueError(
            "Target and prediction arrays must "
            "have the same length."
        )

    matrix = confusion_matrix(
        y_true_array,
        prediction_array,
        labels=[0, 1],
    )

    true_negative = int(
        matrix[0, 0]
    )

    false_positive = int(
        matrix[0, 1]
    )

    false_negative = int(
        matrix[1, 0]
    )

    true_positive = int(
        matrix[1, 1]
    )

    return {
        "Accuracy": accuracy_score(
            y_true_array,
            prediction_array,
        ),
        "BalancedAccuracy": (
            balanced_accuracy_score(
                y_true_array,
                prediction_array,
            )
        ),
        "Precision": precision_score(
            y_true_array,
            prediction_array,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true_array,
            prediction_array,
            zero_division=0,
        ),
        "F1": f1_score(
            y_true_array,
            prediction_array,
            zero_division=0,
        ),
        "TrueNegative": true_negative,
        "FalsePositive": false_positive,
        "FalseNegative": false_negative,
        "TruePositive": true_positive,
        "MismatchCount": int(
            np.sum(
                y_true_array
                != prediction_array
            )
        ),
    }


# ============================================================
# Univariate ROC-AUC Function
# ============================================================

def calculate_univariate_auc(
    feature_values,
    target_values,
):
    """
    Calculate the predictive ROC-AUC of one feature.

    The function checks both possible directions:

    Higher values -> churn
    Lower values  -> churn
    """

    feature_array = np.asarray(
        feature_values,
        dtype=float,
    )

    target_array = np.asarray(
        target_values,
        dtype=int,
    )

    if np.unique(
        feature_array
    ).size < 2:

        return {
            "RawAUC": np.nan,
            "PredictiveAUC": np.nan,
            "AUCDirection": "Constant feature",
        }

    raw_auc = roc_auc_score(
        target_array,
        feature_array,
    )

    if raw_auc >= 0.50:

        predictive_auc = raw_auc

        auc_direction = (
            "Higher values indicate churn"
        )

    else:

        predictive_auc = (
            1.0 - raw_auc
        )

        auc_direction = (
            "Lower values indicate churn"
        )

    return {
        "RawAUC": float(
            raw_auc
        ),
        "PredictiveAUC": float(
            predictive_auc
        ),
        "AUCDirection": (
            auc_direction
        ),
    }


# ============================================================
# Efficient Threshold Search Function
# ============================================================

def find_best_threshold(
    feature_values,
    target_values,
    feature_name,
    return_all_thresholds=False,
):
    """
    Find the single threshold that best reconstructs churn.

    Both rules are tested:

    1. feature > threshold predicts churn
    2. feature <= threshold predicts churn

    Thresholds are placed between consecutive unique values.
    """

    feature_array = np.asarray(
        feature_values,
        dtype=float,
    )

    target_array = np.asarray(
        target_values,
        dtype=int,
    )

    if len(feature_array) != len(
        target_array
    ):
        raise ValueError(
            f"Length mismatch for "
            f"feature '{feature_name}'."
        )

    valid_mask = (
        np.isfinite(
            feature_array
        )
    )

    feature_array = feature_array[
        valid_mask
    ]

    target_array = target_array[
        valid_mask
    ]

    if len(feature_array) == 0:
        raise ValueError(
            f"Feature '{feature_name}' "
            "contains no valid values."
        )

    sorted_indices = np.argsort(
        feature_array,
        kind="mergesort",
    )

    sorted_feature = feature_array[
        sorted_indices
    ]

    sorted_target = target_array[
        sorted_indices
    ]

    total_samples = len(
        sorted_feature
    )

    total_positive = int(
        sorted_target.sum()
    )

    total_negative = int(
        total_samples
        - total_positive
    )

    cumulative_positive = np.concatenate(
        [
            np.array(
                [0],
                dtype=int,
            ),
            np.cumsum(
                sorted_target
            ),
        ]
    )

    cumulative_negative = np.arange(
        total_samples + 1
    ) - cumulative_positive

    # Split position k means:
    # first k observations are <= threshold
    # remaining observations are > threshold.

    valid_split_positions = [0]

    for split_position in range(
        1,
        total_samples,
    ):

        if (
            sorted_feature[
                split_position - 1
            ]
            < sorted_feature[
                split_position
            ]
        ):

            valid_split_positions.append(
                split_position
            )

    valid_split_positions.append(
        total_samples
    )

    threshold_records = []

    for split_position in (
        valid_split_positions
    ):

        if split_position == 0:

            threshold = float(
                np.nextafter(
                    sorted_feature[0],
                    -np.inf,
                )
            )

        elif split_position == total_samples:

            threshold = float(
                sorted_feature[-1]
            )

        else:

            lower_value = float(
                sorted_feature[
                    split_position - 1
                ]
            )

            upper_value = float(
                sorted_feature[
                    split_position
                ]
            )

            threshold = (
                lower_value
                + (
                    upper_value
                    - lower_value
                )
                / 2.0
            )

        positive_left = int(
            cumulative_positive[
                split_position
            ]
        )

        negative_left = int(
            cumulative_negative[
                split_position
            ]
        )

        positive_right = int(
            total_positive
            - positive_left
        )

        negative_right = int(
            total_negative
            - negative_left
        )

        # ----------------------------------------------------
        # Rule A:
        # feature > threshold predicts churn
        # ----------------------------------------------------

        rule_a_true_positive = (
            positive_right
        )

        rule_a_false_positive = (
            negative_right
        )

        rule_a_false_negative = (
            positive_left
        )

        rule_a_true_negative = (
            negative_left
        )

        rule_a_predictions = (
            feature_array
            > threshold
        ).astype(int)

        rule_a_metrics = (
            calculate_binary_metrics(
                target_array,
                rule_a_predictions,
            )
        )

        threshold_records.append(
            {
                "Feature": feature_name,
                "Threshold": threshold,
                "Rule": (
                    "Feature > threshold"
                ),
                "Accuracy": (
                    rule_a_metrics[
                        "Accuracy"
                    ]
                ),
                "BalancedAccuracy": (
                    rule_a_metrics[
                        "BalancedAccuracy"
                    ]
                ),
                "Precision": (
                    rule_a_metrics[
                        "Precision"
                    ]
                ),
                "Recall": (
                    rule_a_metrics[
                        "Recall"
                    ]
                ),
                "F1": (
                    rule_a_metrics[
                        "F1"
                    ]
                ),
                "TrueNegative": (
                    rule_a_true_negative
                ),
                "FalsePositive": (
                    rule_a_false_positive
                ),
                "FalseNegative": (
                    rule_a_false_negative
                ),
                "TruePositive": (
                    rule_a_true_positive
                ),
                "MismatchCount": (
                    rule_a_metrics[
                        "MismatchCount"
                    ]
                ),
                "MatchRate": (
                    rule_a_metrics[
                        "Accuracy"
                    ]
                ),
            }
        )

        # ----------------------------------------------------
        # Rule B:
        # feature <= threshold predicts churn
        # ----------------------------------------------------

        rule_b_true_positive = (
            positive_left
        )

        rule_b_false_positive = (
            negative_left
        )

        rule_b_false_negative = (
            positive_right
        )

        rule_b_true_negative = (
            negative_right
        )

        rule_b_predictions = (
            feature_array
            <= threshold
        ).astype(int)

        rule_b_metrics = (
            calculate_binary_metrics(
                target_array,
                rule_b_predictions,
            )
        )

        threshold_records.append(
            {
                "Feature": feature_name,
                "Threshold": threshold,
                "Rule": (
                    "Feature <= threshold"
                ),
                "Accuracy": (
                    rule_b_metrics[
                        "Accuracy"
                    ]
                ),
                "BalancedAccuracy": (
                    rule_b_metrics[
                        "BalancedAccuracy"
                    ]
                ),
                "Precision": (
                    rule_b_metrics[
                        "Precision"
                    ]
                ),
                "Recall": (
                    rule_b_metrics[
                        "Recall"
                    ]
                ),
                "F1": (
                    rule_b_metrics[
                        "F1"
                    ]
                ),
                "TrueNegative": (
                    rule_b_true_negative
                ),
                "FalsePositive": (
                    rule_b_false_positive
                ),
                "FalseNegative": (
                    rule_b_false_negative
                ),
                "TruePositive": (
                    rule_b_true_positive
                ),
                "MismatchCount": (
                    rule_b_metrics[
                        "MismatchCount"
                    ]
                ),
                "MatchRate": (
                    rule_b_metrics[
                        "Accuracy"
                    ]
                ),
            }
        )

    threshold_results = pd.DataFrame(
        threshold_records
    )

    threshold_results.sort_values(
        by=[
            "Accuracy",
            "BalancedAccuracy",
            "F1",
        ],
        ascending=[
            False,
            False,
            False,
        ],
        inplace=True,
    )

    threshold_results.reset_index(
        drop=True,
        inplace=True,
    )

    best_result = (
        threshold_results.iloc[0]
        .to_dict()
    )

    if return_all_thresholds:

        return (
            best_result,
            threshold_results,
        )

    return best_result


# ============================================================
# Binary Direct-Match Function
# ============================================================

def calculate_binary_feature_match(
    feature_values,
    target_values,
):
    """
    Test whether a two-valued feature directly reproduces
    the churn target.
    """

    feature_series = pd.Series(
        feature_values
    )

    unique_values = sorted(
        feature_series.dropna().unique()
    )

    if len(unique_values) != 2:

        return {
            "BinaryDirectMatchRate": np.nan,
            "BinaryInverseMatchRate": np.nan,
            "BestBinaryMatchRate": np.nan,
            "BinaryMatchDirection": (
                "Not a binary feature"
            ),
        }

    lower_value = unique_values[0]
    upper_value = unique_values[1]

    mapped_values = (
        feature_series
        .map(
            {
                lower_value: 0,
                upper_value: 1,
            }
        )
        .astype(int)
        .to_numpy()
    )

    target_array = np.asarray(
        target_values,
        dtype=int,
    )

    direct_match_rate = float(
        np.mean(
            mapped_values
            == target_array
        )
    )

    inverse_values = (
        1 - mapped_values
    )

    inverse_match_rate = float(
        np.mean(
            inverse_values
            == target_array
        )
    )

    if direct_match_rate >= (
        inverse_match_rate
    ):

        best_match_rate = (
            direct_match_rate
        )

        match_direction = (
            f"{lower_value}->0, "
            f"{upper_value}->1"
        )

    else:

        best_match_rate = (
            inverse_match_rate
        )

        match_direction = (
            f"{lower_value}->1, "
            f"{upper_value}->0"
        )

    return {
        "BinaryDirectMatchRate": (
            direct_match_rate
        ),
        "BinaryInverseMatchRate": (
            inverse_match_rate
        ),
        "BestBinaryMatchRate": (
            best_match_rate
        ),
        "BinaryMatchDirection": (
            match_direction
        ),
    }


# ============================================================
# Feature Leakage Risk Classification
# ============================================================

def classify_feature_leakage_risk(
    predictive_auc,
    threshold_match_rate,
    absolute_correlation,
    binary_match_rate,
):
    """
    Assign a statistical leakage-risk level.

    This is a screening result, not final proof of leakage.
    Source-code and temporal-window evidence are still needed.
    """

    valid_binary_match = (
        0.0
        if pd.isna(binary_match_rate)
        else float(binary_match_rate)
    )

    maximum_signal = max(
        float(
            predictive_auc
            if not pd.isna(
                predictive_auc
            )
            else 0.0
        ),
        float(
            threshold_match_rate
            if not pd.isna(
                threshold_match_rate
            )
            else 0.0
        ),
        float(
            absolute_correlation
            if not pd.isna(
                absolute_correlation
            )
            else 0.0
        ),
        valid_binary_match,
    )

    if (
        threshold_match_rate
        >= 0.995
        or valid_binary_match
        >= 0.995
    ):

        return (
            "CRITICAL",
            "Feature nearly or exactly reconstructs target",
        )

    if (
        predictive_auc >= 0.95
        or threshold_match_rate >= 0.95
        or absolute_correlation >= 0.90
        or valid_binary_match >= 0.95
    ):

        return (
            "HIGH",
            "Feature has near-deterministic target relationship",
        )

    if (
        predictive_auc >= 0.85
        or threshold_match_rate >= 0.85
        or absolute_correlation >= 0.70
        or valid_binary_match >= 0.85
    ):

        return (
            "MODERATE",
            "Feature has unusually strong univariate signal",
        )

    if maximum_signal >= 0.70:

        return (
            "LOW-MODERATE",
            "Feature has meaningful but non-deterministic signal",
        )

    return (
        "LOW",
        "No near-deterministic statistical relationship detected",
    )


# ============================================================
# Audit Every Numeric Feature
# ============================================================

print("\n" + "=" * 70)
print("UNIVARIATE FEATURE LEAKAGE TESTS")
print("=" * 70)

feature_audit_records = []

recency_threshold_results = None

for feature in numeric_features:

    feature_values = (
        numeric_df[
            feature
        ]
    )

    target_values = (
        df[TARGET]
    )

    auc_results = (
        calculate_univariate_auc(
            feature_values,
            target_values,
        )
    )

    if feature == "Recency":

        (
            threshold_result,
            recency_threshold_results,
        ) = find_best_threshold(
            feature_values=feature_values,
            target_values=target_values,
            feature_name=feature,
            return_all_thresholds=True,
        )

    else:

        threshold_result = (
            find_best_threshold(
                feature_values=feature_values,
                target_values=target_values,
                feature_name=feature,
                return_all_thresholds=False,
            )
        )

    binary_results = (
        calculate_binary_feature_match(
            feature_values,
            target_values,
        )
    )

    non_churn_values = (
        feature_values[
            df[TARGET] == 0
        ]
    )

    churn_values = (
        feature_values[
            df[TARGET] == 1
        ]
    )

    non_churn_mean = float(
        non_churn_values.mean()
    )

    churn_mean = float(
        churn_values.mean()
    )

    non_churn_median = float(
        non_churn_values.median()
    )

    churn_median = float(
        churn_values.median()
    )

    feature_standard_deviation = float(
        feature_values.std(
            ddof=0
        )
    )

    if feature_standard_deviation > 0:

        standardized_mean_difference = (
            churn_mean
            - non_churn_mean
        ) / feature_standard_deviation

    else:

        standardized_mean_difference = 0.0

    correlation_match = (
        correlation_df[
            correlation_df[
                "Feature"
            ] == feature
        ]
    )

    if correlation_match.empty:

        pearson_correlation = np.nan
        absolute_correlation = np.nan

    else:

        pearson_correlation = float(
            correlation_match.iloc[0][
                "PearsonCorrelation"
            ]
        )

        absolute_correlation = float(
            correlation_match.iloc[0][
                "AbsoluteCorrelation"
            ]
        )

    (
        leakage_risk,
        leakage_reason,
    ) = classify_feature_leakage_risk(
        predictive_auc=auc_results[
            "PredictiveAUC"
        ],
        threshold_match_rate=threshold_result[
            "MatchRate"
        ],
        absolute_correlation=absolute_correlation,
        binary_match_rate=binary_results[
            "BestBinaryMatchRate"
        ],
    )

    exact_threshold_reconstruction = bool(
        np.isclose(
            threshold_result[
                "MatchRate"
            ],
            1.0,
        )
    )

    near_exact_reconstruction = bool(
        threshold_result[
            "MatchRate"
        ]
        >= 0.95
    )

    feature_audit_records.append(
        {
            "Feature": feature,
            "UniqueValues": int(
                feature_values.nunique()
            ),
            "NonChurnMean": (
                non_churn_mean
            ),
            "ChurnMean": churn_mean,
            "NonChurnMedian": (
                non_churn_median
            ),
            "ChurnMedian": (
                churn_median
            ),
            "StandardizedMeanDifference": (
                standardized_mean_difference
            ),
            "PearsonCorrelation": (
                pearson_correlation
            ),
            "AbsoluteCorrelation": (
                absolute_correlation
            ),
            "RawUnivariateAUC": (
                auc_results[
                    "RawAUC"
                ]
            ),
            "PredictiveUnivariateAUC": (
                auc_results[
                    "PredictiveAUC"
                ]
            ),
            "AUCDirection": (
                auc_results[
                    "AUCDirection"
                ]
            ),
            "BestThreshold": (
                threshold_result[
                    "Threshold"
                ]
            ),
            "BestThresholdRule": (
                threshold_result[
                    "Rule"
                ]
            ),
            "ThresholdMatchRate": (
                threshold_result[
                    "MatchRate"
                ]
            ),
            "ThresholdBalancedAccuracy": (
                threshold_result[
                    "BalancedAccuracy"
                ]
            ),
            "ThresholdPrecision": (
                threshold_result[
                    "Precision"
                ]
            ),
            "ThresholdRecall": (
                threshold_result[
                    "Recall"
                ]
            ),
            "ThresholdF1": (
                threshold_result[
                    "F1"
                ]
            ),
            "ThresholdMismatchCount": int(
                threshold_result[
                    "MismatchCount"
                ]
            ),
            "ExactThresholdReconstruction": (
                exact_threshold_reconstruction
            ),
            "NearExactReconstruction": (
                near_exact_reconstruction
            ),
            "BestBinaryMatchRate": (
                binary_results[
                    "BestBinaryMatchRate"
                ]
            ),
            "BinaryMatchDirection": (
                binary_results[
                    "BinaryMatchDirection"
                ]
            ),
            "StatisticalLeakageRisk": (
                leakage_risk
            ),
            "LeakageRiskReason": (
                leakage_reason
            ),
        }
    )

    print(
        f"{feature:<24} "
        f"AUC: "
        f"{auc_results['PredictiveAUC']:.4f} | "
        f"Best threshold match: "
        f"{threshold_result['MatchRate']:.4f} | "
        f"Risk: {leakage_risk}"
    )

# ============================================================
# Create Feature Audit Table
# ============================================================

feature_audit_df = pd.DataFrame(
    feature_audit_records
)

risk_order = {
    "CRITICAL": 5,
    "HIGH": 4,
    "MODERATE": 3,
    "LOW-MODERATE": 2,
    "LOW": 1,
}

feature_audit_df[
    "RiskOrder"
] = feature_audit_df[
    "StatisticalLeakageRisk"
].map(
    risk_order
)

feature_audit_df.sort_values(
    by=[
        "RiskOrder",
        "ThresholdMatchRate",
        "PredictiveUnivariateAUC",
    ],
    ascending=[
        False,
        False,
        False,
    ],
    inplace=True,
)

feature_audit_df.drop(
    columns=[
        "RiskOrder",
    ],
    inplace=True,
)

feature_audit_df.reset_index(
    drop=True,
    inplace=True,
)

feature_audit_df.to_csv(
    FEATURE_AUDIT_PATH,
    index=False,
)

print(
    f"\nFeature audit saved: "
    f"{FEATURE_AUDIT_PATH}"
)

# ============================================================
# Save Full Recency Threshold Audit
# ============================================================

if recency_threshold_results is not None:

    recency_threshold_results.to_csv(
        THRESHOLD_AUDIT_PATH,
        index=False,
    )

    print(
        f"Recency threshold audit saved: "
        f"{THRESHOLD_AUDIT_PATH}"
    )

# ============================================================
# Detailed Recency Audit
# ============================================================

recency_audit = (
    feature_audit_df[
        feature_audit_df[
            "Feature"
        ] == "Recency"
    ]
)

print("\n" + "=" * 70)
print("RECENCY LEAKAGE AUDIT")
print("=" * 70)

if recency_audit.empty:

    print(
        "Recency was not found in the dataset."
    )

    recency_summary = None

else:

    recency_summary = (
        recency_audit.iloc[0]
    )

    print(
        f"Univariate ROC-AUC       : "
        f"{recency_summary['PredictiveUnivariateAUC']:.4f}"
    )

    print(
        f"Best Threshold           : "
        f"{recency_summary['BestThreshold']:.4f}"
    )

    print(
        f"Best Rule                : "
        f"{recency_summary['BestThresholdRule']}"
    )

    print(
        f"Target Match Rate        : "
        f"{recency_summary['ThresholdMatchRate']:.4f}"
    )

    print(
        f"Mismatch Count           : "
        f"{int(recency_summary['ThresholdMismatchCount'])}"
    )

    print(
        f"Threshold Precision      : "
        f"{recency_summary['ThresholdPrecision']:.4f}"
    )

    print(
        f"Threshold Recall         : "
        f"{recency_summary['ThresholdRecall']:.4f}"
    )

    print(
        f"Threshold F1             : "
        f"{recency_summary['ThresholdF1']:.4f}"
    )

    print(
        f"Exact Reconstruction     : "
        f"{recency_summary['ExactThresholdReconstruction']}"
    )

    print(
        f"Near-Exact Reconstruction: "
        f"{recency_summary['NearExactReconstruction']}"
    )

    print(
        f"Statistical Risk Level   : "
        f"{recency_summary['StatisticalLeakageRisk']}"
    )

    print(
        f"Risk Reason              : "
        f"{recency_summary['LeakageRiskReason']}"
    )

# ============================================================
# Display Highest-Risk Features
# ============================================================

print("\n" + "=" * 70)
print("HIGHEST STATISTICAL LEAKAGE RISKS")
print("=" * 70)

display_columns = [
    "Feature",
    "PredictiveUnivariateAUC",
    "ThresholdMatchRate",
    "ThresholdMismatchCount",
    "StatisticalLeakageRisk",
]

for _, row in (
    feature_audit_df[
        display_columns
    ]
    .head(10)
    .iterrows()
):

    print(
        f"{row['Feature']:<24} "
        f"AUC: "
        f"{row['PredictiveUnivariateAUC']:.4f} | "
        f"Match: "
        f"{row['ThresholdMatchRate']:.4f} | "
        f"Errors: "
        f"{int(row['ThresholdMismatchCount'])} | "
        f"Risk: "
        f"{row['StatisticalLeakageRisk']}"
    )

# ============================================================
# Statistical Audit Summary
# ============================================================

critical_feature_count = int(
    (
        feature_audit_df[
            "StatisticalLeakageRisk"
        ]
        == "CRITICAL"
    ).sum()
)

high_feature_count = int(
    (
        feature_audit_df[
            "StatisticalLeakageRisk"
        ]
        == "HIGH"
    ).sum()
)

moderate_feature_count = int(
    (
        feature_audit_df[
            "StatisticalLeakageRisk"
        ]
        == "MODERATE"
    ).sum()
)

print("\n" + "=" * 70)
print("STATISTICAL AUDIT SUMMARY")
print("=" * 70)

print(
    f"Features Audited          : "
    f"{len(feature_audit_df)}"
)

print(
    f"Critical-Risk Features    : "
    f"{critical_feature_count}"
)

print(
    f"High-Risk Features        : "
    f"{high_feature_count}"
)

print(
    f"Moderate-Risk Features    : "
    f"{moderate_feature_count}"
)

if critical_feature_count > 0:

    print(
        "\nA feature nearly or exactly reconstructs "
        "the churn target."
    )

    print(
        "This is strong statistical evidence of "
        "possible target leakage."
    )

elif high_feature_count > 0:

    print(
        "\nAt least one feature has an unusually "
        "strong relationship with churn."
    )

    print(
        "Review the churn-label definition and "
        "temporal feature window carefully."
    )

else:

    print(
        "\nNo feature was found to reconstruct "
        "the target nearly perfectly."
    )

    print(
        "Temporal leakage may still exist and "
        "must be checked from source logic."
    )

print(
    "\nProceeding to final source-code evidence, "
    "temporal leakage decision, and audit report..."
)

# ============================================================
# Final Output Configuration
# ============================================================

SUMMARY_JSON_PATH = (
    REPORT_DIRECTORY
    / "target_leakage_audit_summary.json"
)

CURRENT_AUDIT_FILE = Path(
    __file__
).resolve()

CURRENT_AUDIT_RELATIVE_PATH = str(
    CURRENT_AUDIT_FILE.relative_to(
        PROJECT_ROOT
    )
)

# ============================================================
# Remove Matches Generated by This Audit Script
# ============================================================

# The source scan from Part 1 also scans this audit file.
# Those matches must be excluded because this script contains
# words such as Churn, Recency, threshold, and future window.

if source_matches_df.empty:

    filtered_source_matches_df = pd.DataFrame(
        columns=[
            "File",
            "LineNumber",
            "Pattern",
            "SourceLine",
            "Context",
        ]
    )

else:

    source_matches_df = (
        source_matches_df.copy()
    )

    def normalize_path_string(
        path_value,
    ):
        """
        Standardize slash direction for reliable path comparison.
        """

        return str(
            path_value
        ).replace(
            "\\",
            "/",
        )

    normalized_audit_path = (
        normalize_path_string(
            CURRENT_AUDIT_RELATIVE_PATH
        )
    )

    source_matches_df[
        "NormalizedFile"
    ] = source_matches_df[
        "File"
    ].apply(
        normalize_path_string
    )

    filtered_source_matches_df = (
        source_matches_df[
            source_matches_df[
                "NormalizedFile"
            ]
            != normalized_audit_path
        ]
        .copy()
    )

    filtered_source_matches_df.drop(
        columns=[
            "NormalizedFile",
        ],
        inplace=True,
        errors="ignore",
    )

print("\nFiltering source-code matches...")

print(
    f"Matches Before Filtering : "
    f"{len(source_matches_df)}"
)

print(
    f"Matches After Filtering  : "
    f"{len(filtered_source_matches_df)}"
)

# ============================================================
# Add Source Context Around Each Match
# ============================================================

def extract_source_context(
    relative_file,
    line_number,
    context_radius=2,
):
    """
    Return the matched source line with surrounding context.
    """

    absolute_path = (
        PROJECT_ROOT
        / relative_file
    )

    if not absolute_path.exists():
        return ""

    source_text = read_source_text(
        absolute_path
    )

    if not source_text:
        return ""

    source_lines = (
        source_text.splitlines()
    )

    line_index = int(
        line_number
    ) - 1

    start_index = max(
        0,
        line_index - context_radius,
    )

    end_index = min(
        len(source_lines),
        line_index
        + context_radius
        + 1,
    )

    context_parts = []

    for current_index in range(
        start_index,
        end_index,
    ):

        marker = (
            ">>"
            if current_index
            == line_index
            else "  "
        )

        context_parts.append(
            f"{marker} "
            f"{current_index + 1}: "
            f"{source_lines[current_index].strip()}"
        )

    return " || ".join(
        context_parts
    )


if not filtered_source_matches_df.empty:

    filtered_source_matches_df[
        "Context"
    ] = filtered_source_matches_df.apply(
        lambda row: extract_source_context(
            relative_file=row["File"],
            line_number=row["LineNumber"],
            context_radius=2,
        ),
        axis=1,
    )

else:

    filtered_source_matches_df[
        "Context"
    ] = pd.Series(
        dtype=str
    )

# Save filtered source evidence even when no matches exist.
filtered_source_matches_df.to_csv(
    SOURCE_MATCHES_PATH,
    index=False,
)

print(
    f"Filtered source evidence saved: "
    f"{SOURCE_MATCHES_PATH}"
)

# ============================================================
# Classify Source-Code Evidence
# ============================================================

def source_line_contains_comparison(
    source_line,
):
    """
    Check whether a line contains a comparison operator.
    """

    comparison_pattern = re.compile(
        r"(>=|<=|>|<)"
    )

    return bool(
        comparison_pattern.search(
            str(source_line)
        )
    )


def source_line_mentions_recency_and_churn(
    source_line,
):
    """
    Check whether a source line refers to both Recency and Churn.
    """

    normalized_line = str(
        source_line
    ).lower()

    return (
        "recency"
        in normalized_line
        and "churn"
        in normalized_line
    )


if filtered_source_matches_df.empty:

    direct_assignment_matches = (
        filtered_source_matches_df.copy()
    )

    recency_rule_matches = (
        filtered_source_matches_df.copy()
    )

    future_window_matches = (
        filtered_source_matches_df.copy()
    )

    no_purchase_matches = (
        filtered_source_matches_df.copy()
    )

else:

    direct_assignment_matches = (
        filtered_source_matches_df[
            filtered_source_matches_df[
                "Pattern"
            ]
            == "Direct churn assignment"
        ]
        .copy()
    )

    recency_rule_mask = (
        filtered_source_matches_df[
            "SourceLine"
        ].apply(
            source_line_mentions_recency_and_churn
        )
        |
        (
            filtered_source_matches_df[
                "Pattern"
            ]
            == "Recency threshold comparison"
        )
    )

    recency_rule_matches = (
        filtered_source_matches_df[
            recency_rule_mask
        ]
        .copy()
    )

    future_window_matches = (
        filtered_source_matches_df[
            filtered_source_matches_df[
                "Pattern"
            ].isin(
                [
                    "Future observation window",
                    "Shift or forward target",
                ]
            )
        ]
        .copy()
    )

    no_purchase_matches = (
        filtered_source_matches_df[
            filtered_source_matches_df[
                "Pattern"
            ]
            == "No-purchase logic"
        ]
        .copy()
    )

direct_assignment_count = int(
    len(
        direct_assignment_matches
    )
)

recency_rule_count = int(
    len(
        recency_rule_matches
    )
)

future_window_count = int(
    len(
        future_window_matches
    )
)

no_purchase_logic_count = int(
    len(
        no_purchase_matches
    )
)

# ============================================================
# Extract Recency Statistical Evidence
# ============================================================

if recency_summary is None:

    recency_auc = np.nan
    recency_match_rate = np.nan
    recency_mismatch_count = None
    recency_best_threshold = np.nan
    recency_best_rule = "Unavailable"
    recency_risk_level = "UNKNOWN"
    recency_exact_reconstruction = False
    recency_near_exact_reconstruction = False

else:

    recency_auc = float(
        recency_summary[
            "PredictiveUnivariateAUC"
        ]
    )

    recency_match_rate = float(
        recency_summary[
            "ThresholdMatchRate"
        ]
    )

    recency_mismatch_count = int(
        recency_summary[
            "ThresholdMismatchCount"
        ]
    )

    recency_best_threshold = float(
        recency_summary[
            "BestThreshold"
        ]
    )

    recency_best_rule = str(
        recency_summary[
            "BestThresholdRule"
        ]
    )

    recency_risk_level = str(
        recency_summary[
            "StatisticalLeakageRisk"
        ]
    )

    recency_exact_reconstruction = bool(
        recency_summary[
            "ExactThresholdReconstruction"
        ]
    )

    recency_near_exact_reconstruction = bool(
        recency_summary[
            "NearExactReconstruction"
        ]
    )

# ============================================================
# Determine Temporal-Design Evidence
# ============================================================

has_direct_churn_assignment = (
    direct_assignment_count > 0
)

has_recency_threshold_evidence = (
    recency_rule_count > 0
)

has_future_window_evidence = (
    future_window_count > 0
)

has_no_purchase_logic = (
    no_purchase_logic_count > 0
)

# A no-purchase rule alone is not enough to establish a safe
# temporal target. The features must be created before a cutoff,
# and the target must be measured in a later observation window.

has_temporal_target_evidence = bool(
    has_future_window_evidence
)

# ============================================================
# Overall Leakage-Risk Decision
# ============================================================

if (
    recency_exact_reconstruction
    and has_recency_threshold_evidence
):

    overall_leakage_risk = (
        "VERY HIGH"
    )

    overall_leakage_status = (
        "The Recency feature exactly reconstructs the "
        "target and source-code evidence contains a "
        "Recency-based churn rule."
    )

    recommended_action = (
        "Do not use the current churn dataset for the final "
        "production evaluation. Rebuild the target using a "
        "historical feature window and a separate future "
        "observation window."
    )

elif (
    recency_near_exact_reconstruction
    or critical_feature_count > 0
):

    overall_leakage_risk = (
        "VERY HIGH"
    )

    overall_leakage_status = (
        "At least one feature nearly reconstructs the target. "
        "This is strong statistical evidence of possible "
        "target leakage."
    )

    recommended_action = (
        "Inspect the exact churn-label construction and rebuild "
        "the dataset using time-separated feature and target "
        "windows before claiming final model performance."
    )

elif (
    high_feature_count > 0
    and not has_temporal_target_evidence
):

    overall_leakage_risk = (
        "HIGH"
    )

    overall_leakage_status = (
        "One or more features have near-deterministic target "
        "relationships, while no clear future observation-window "
        "logic was detected."
    )

    recommended_action = (
        "Perform a manual source-code review and create a "
        "future-window churn target before further tuning."
    )

elif (
    has_direct_churn_assignment
    and has_recency_threshold_evidence
    and not has_temporal_target_evidence
):

    overall_leakage_risk = (
        "HIGH"
    )

    overall_leakage_status = (
        "Source-code patterns suggest that churn may be assigned "
        "using Recency or another same-window threshold."
    )

    recommended_action = (
        "Replace the same-window threshold label with a future "
        "purchase or inactivity label."
    )

elif moderate_feature_count > 0:

    overall_leakage_risk = (
        "MODERATE"
    )

    overall_leakage_status = (
        "Some features have unusually strong univariate signal, "
        "but no near-perfect target reconstruction was detected."
    )

    recommended_action = (
        "Confirm that every feature is calculated strictly before "
        "the churn observation window. Use a time-based split for "
        "model validation."
    )

else:

    overall_leakage_risk = (
        "LOW"
    )

    overall_leakage_status = (
        "No near-deterministic statistical relationship was "
        "detected in the available processed features."
    )

    recommended_action = (
        "Continue with a manual temporal audit because statistical "
        "tests alone cannot rule out future-data leakage."
    )

# ============================================================
# Determine Audit Outcome
# ============================================================

if overall_leakage_risk in [
    "VERY HIGH",
    "HIGH",
]:

    audit_outcome = (
        "CURRENT DATASET REQUIRES CORRECTION"
    )

elif overall_leakage_risk == (
    "MODERATE"
):

    audit_outcome = (
        "MANUAL TEMPORAL REVIEW REQUIRED"
    )

else:

    audit_outcome = (
        "NO CRITICAL STATISTICAL LEAKAGE DETECTED"
    )

# ============================================================
# Display Source Evidence Summary
# ============================================================

print("\n" + "=" * 70)
print("SOURCE-CODE EVIDENCE SUMMARY")
print("=" * 70)

print(
    f"Direct Churn Assignments : "
    f"{direct_assignment_count}"
)

print(
    f"Recency Rule Matches     : "
    f"{recency_rule_count}"
)

print(
    f"Future-Window Matches    : "
    f"{future_window_count}"
)

print(
    f"No-Purchase Matches      : "
    f"{no_purchase_logic_count}"
)

print(
    f"Temporal Target Evidence : "
    f"{has_temporal_target_evidence}"
)

if not filtered_source_matches_df.empty:

    print(
        "\nDetected Source Evidence"
    )

    print("-" * 70)

    for _, row in (
        filtered_source_matches_df
        .head(20)
        .iterrows()
    ):

        print(
            f"{row['File']}:"
            f"{int(row['LineNumber'])} | "
            f"{row['Pattern']} | "
            f"{row['SourceLine']}"
        )

else:

    print(
        "\nNo relevant churn-label source "
        "matches were detected automatically."
    )

# ============================================================
# Display Overall Leakage Decision
# ============================================================

print("\n" + "=" * 70)
print("OVERALL TARGET-LEAKAGE DECISION")
print("=" * 70)

print(
    f"Overall Leakage Risk : "
    f"{overall_leakage_risk}"
)

print(
    f"Audit Outcome        : "
    f"{audit_outcome}"
)

print(
    f"\nAssessment:\n"
    f"{overall_leakage_status}"
)

print(
    f"\nRecommended Action:\n"
    f"{recommended_action}"
)

# ============================================================
# Save Machine-Readable Audit Summary
# ============================================================

summary_data = {
    "project": "RetailPulse",
    "function": "F04 Customer Churn Prediction",
    "dataset_path": str(
        DATA_PATH
    ),
    "dataset_rows": int(
        len(df)
    ),
    "dataset_columns": int(
        df.shape[1]
    ),
    "features_audited": int(
        len(feature_audit_df)
    ),
    "critical_risk_features": int(
        critical_feature_count
    ),
    "high_risk_features": int(
        high_feature_count
    ),
    "moderate_risk_features": int(
        moderate_feature_count
    ),
    "recency_present": bool(
        recency_summary
        is not None
    ),
    "recency_univariate_auc": (
        None
        if pd.isna(recency_auc)
        else float(recency_auc)
    ),
    "recency_threshold_match_rate": (
        None
        if pd.isna(
            recency_match_rate
        )
        else float(
            recency_match_rate
        )
    ),
    "recency_threshold_mismatch_count": (
        recency_mismatch_count
    ),
    "recency_best_threshold": (
        None
        if pd.isna(
            recency_best_threshold
        )
        else float(
            recency_best_threshold
        )
    ),
    "recency_best_rule": (
        recency_best_rule
    ),
    "recency_statistical_risk": (
        recency_risk_level
    ),
    "recency_exact_reconstruction": bool(
        recency_exact_reconstruction
    ),
    "recency_near_exact_reconstruction": bool(
        recency_near_exact_reconstruction
    ),
    "direct_churn_assignment_matches": int(
        direct_assignment_count
    ),
    "recency_rule_matches": int(
        recency_rule_count
    ),
    "future_window_matches": int(
        future_window_count
    ),
    "no_purchase_logic_matches": int(
        no_purchase_logic_count
    ),
    "temporal_target_evidence_detected": bool(
        has_temporal_target_evidence
    ),
    "overall_leakage_risk": (
        overall_leakage_risk
    ),
    "audit_outcome": (
        audit_outcome
    ),
    "assessment": (
        overall_leakage_status
    ),
    "recommended_action": (
        recommended_action
    ),
}

with open(
    SUMMARY_JSON_PATH,
    "w",
    encoding="utf-8",
) as summary_file:

    json.dump(
        summary_data,
        summary_file,
        indent=4,
    )

print(
    f"\nAudit summary JSON saved: "
    f"{SUMMARY_JSON_PATH}"
)

# ============================================================
# Build Correct Temporal Target Guidance
# ============================================================

temporal_target_guidance = """
Recommended leakage-safe churn design

1. Select a historical feature cutoff date.
2. Calculate Recency, Frequency, Monetary, quantity, and lifetime
   using transactions occurring on or before the cutoff date.
3. Create a future observation window after the cutoff date,
   such as the following 60 or 90 days.
4. Set Churn = 1 when the customer makes no purchase during the
   future observation window.
5. Set Churn = 0 when the customer makes at least one purchase
   during the future observation window.
6. Exclude customers without enough future observation time.
7. Split training and testing chronologically rather than using
   a random split when possible.
8. Fit preprocessing, feature selection, and tuning using only
   the training period.
""".strip()

# ============================================================
# Write Complete Leakage Audit Report
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as report:

    report.write("RetailPulse\n")

    report.write(
        "F04 Customer Churn Prediction\n"
    )

    report.write(
        "Target Leakage Audit Report\n"
    )

    report.write("=" * 78)
    report.write("\n\n")

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    report.write(
        "Dataset Information\n"
    )

    report.write("-" * 78)
    report.write("\n")

    report.write(
        f"Dataset Path: {DATA_PATH}\n"
    )

    report.write(
        f"Dataset Shape: {df.shape}\n"
    )

    report.write(
        f"Customer Count: {len(df)}\n"
    )

    report.write(
        f"Candidate Features: "
        f"{len(candidate_features)}\n"
    )

    report.write(
        f"Numeric Features Audited: "
        f"{len(numeric_features)}\n"
    )

    report.write(
        f"Duplicate Customer IDs: "
        f"{duplicate_customer_count}\n\n"
    )

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    report.write(
        "Target Distribution\n"
    )

    report.write("-" * 78)
    report.write("\n")

    for target_value, count in (
        target_distribution.items()
    ):

        percentage = (
            count
            / len(df)
            * 100
        )

        target_label = (
            "Non-Churn"
            if target_value == 0
            else "Churn"
        )

        report.write(
            f"{target_label}: "
            f"{count} "
            f"({percentage:.2f}%)\n"
        )

    report.write("\n")

    # --------------------------------------------------------
    # Statistical leakage summary
    # --------------------------------------------------------

    report.write(
        "Statistical Leakage Summary\n"
    )

    report.write("-" * 78)
    report.write("\n")

    report.write(
        f"Critical-Risk Features: "
        f"{critical_feature_count}\n"
    )

    report.write(
        f"High-Risk Features: "
        f"{high_feature_count}\n"
    )

    report.write(
        f"Moderate-Risk Features: "
        f"{moderate_feature_count}\n\n"
    )

    report.write(
        feature_audit_df.to_string(
            index=False,
        )
    )

    report.write("\n\n")

    # --------------------------------------------------------
    # Recency audit
    # --------------------------------------------------------

    report.write(
        "Recency Leakage Audit\n"
    )

    report.write("-" * 78)
    report.write("\n")

    if recency_summary is None:

        report.write(
            "Recency was not present in "
            "the audited dataset.\n\n"
        )

    else:

        report.write(
            f"Univariate ROC-AUC: "
            f"{recency_auc:.6f}\n"
        )

        report.write(
            f"Best Threshold: "
            f"{recency_best_threshold:.6f}\n"
        )

        report.write(
            f"Best Rule: "
            f"{recency_best_rule}\n"
        )

        report.write(
            f"Target Match Rate: "
            f"{recency_match_rate:.6f}\n"
        )

        report.write(
            f"Mismatch Count: "
            f"{recency_mismatch_count}\n"
        )

        report.write(
            f"Exact Reconstruction: "
            f"{recency_exact_reconstruction}\n"
        )

        report.write(
            f"Near-Exact Reconstruction: "
            f"{recency_near_exact_reconstruction}\n"
        )

        report.write(
            f"Statistical Risk: "
            f"{recency_risk_level}\n\n"
        )

    # --------------------------------------------------------
    # Source-code evidence
    # --------------------------------------------------------

    report.write(
        "Source-Code Evidence\n"
    )

    report.write("-" * 78)
    report.write("\n")

    report.write(
        f"Direct Churn Assignments: "
        f"{direct_assignment_count}\n"
    )

    report.write(
        f"Recency Rule Matches: "
        f"{recency_rule_count}\n"
    )

    report.write(
        f"Future-Window Matches: "
        f"{future_window_count}\n"
    )

    report.write(
        f"No-Purchase Logic Matches: "
        f"{no_purchase_logic_count}\n"
    )

    report.write(
        f"Temporal Target Evidence Detected: "
        f"{has_temporal_target_evidence}\n\n"
    )

    if filtered_source_matches_df.empty:

        report.write(
            "No source-code matches were detected "
            "after excluding the audit script.\n\n"
        )

    else:

        source_report_columns = [
            "File",
            "LineNumber",
            "Pattern",
            "SourceLine",
            "Context",
        ]

        report.write(
            filtered_source_matches_df[
                source_report_columns
            ].to_string(
                index=False,
            )
        )

        report.write("\n\n")

    # --------------------------------------------------------
    # Overall conclusion
    # --------------------------------------------------------

    report.write(
        "Overall Leakage Assessment\n"
    )

    report.write("-" * 78)
    report.write("\n")

    report.write(
        f"Overall Leakage Risk: "
        f"{overall_leakage_risk}\n"
    )

    report.write(
        f"Audit Outcome: "
        f"{audit_outcome}\n\n"
    )

    report.write(
        f"Assessment:\n"
        f"{overall_leakage_status}\n\n"
    )

    report.write(
        f"Recommended Action:\n"
        f"{recommended_action}\n\n"
    )

    # --------------------------------------------------------
    # Correct target design
    # --------------------------------------------------------

    report.write(
        "Recommended Leakage-Safe Target Design\n"
    )

    report.write("-" * 78)
    report.write("\n")

    report.write(
        temporal_target_guidance
    )

    report.write("\n\n")

    # --------------------------------------------------------
    # Important interpretation note
    # --------------------------------------------------------

    report.write(
        "Interpretation Limitation\n"
    )

    report.write("-" * 78)
    report.write("\n")

    report.write(
        "A strong feature-target relationship does not prove "
        "leakage by itself. Leakage is confirmed by examining "
        "whether the feature uses target-window or future "
        "information, or whether the target was directly derived "
        "from that feature. This audit combines statistical "
        "screening with automated source-code pattern detection, "
        "but manual review remains necessary.\n\n"
    )

    # --------------------------------------------------------
    # Generated files
    # --------------------------------------------------------

    report.write(
        "Generated Files\n"
    )

    report.write("-" * 78)
    report.write("\n")

    report_generated_files = [
        REPORT_PATH,
        SUMMARY_JSON_PATH,
        FEATURE_AUDIT_PATH,
        THRESHOLD_AUDIT_PATH,
        SOURCE_MATCHES_PATH,
    ]

    for generated_file in (
        report_generated_files
    ):

        report.write(
            f"- {generated_file}\n"
        )

print(
    f"Complete leakage audit saved: "
    f"{REPORT_PATH}"
)

# ============================================================
# Final Feature-Risk Summary
# ============================================================

print("\n" + "=" * 70)
print("TOP AUDITED FEATURES")
print("=" * 70)

summary_columns = [
    "Feature",
    "PredictiveUnivariateAUC",
    "ThresholdMatchRate",
    "ThresholdMismatchCount",
    "StatisticalLeakageRisk",
]

for _, row in (
    feature_audit_df[
        summary_columns
    ]
    .head(10)
    .iterrows()
):

    print(
        f"{row['Feature']:<24} "
        f"AUC: "
        f"{row['PredictiveUnivariateAUC']:.4f} | "
        f"Threshold Match: "
        f"{row['ThresholdMatchRate']:.4f} | "
        f"Errors: "
        f"{int(row['ThresholdMismatchCount'])} | "
        f"Risk: "
        f"{row['StatisticalLeakageRisk']}"
    )

# ============================================================
# Generated Files
# ============================================================

generated_files = [
    REPORT_PATH,
    SUMMARY_JSON_PATH,
    FEATURE_AUDIT_PATH,
    SOURCE_MATCHES_PATH,
]

if THRESHOLD_AUDIT_PATH.exists():

    generated_files.append(
        THRESHOLD_AUDIT_PATH
    )

print("\nGenerated Files")
print("-" * 70)

for generated_file in (
    generated_files
):

    print(
        f"✔ {generated_file}"
    )

# ============================================================
# Exact Next Action
# ============================================================

print("\n" + "=" * 70)
print("NEXT ACTION")
print("=" * 70)

if overall_leakage_risk in [
    "VERY HIGH",
    "HIGH",
]:

    print(
        "1. Do not present the current ROC-AUC "
        "as leakage-safe final performance."
    )

    print(
        "2. Review the detected churn-label "
        "creation lines."
    )

    print(
        "3. Rebuild churn using a historical "
        "feature cutoff and future observation window."
    )

    print(
        "4. Retrain XGBoost using a chronological "
        "train/test split."
    )

    print(
        "5. Recalculate ROC-AUC, Precision@Top20, "
        "and SHAP explanations."
    )

elif overall_leakage_risk == (
    "MODERATE"
):

    print(
        "1. Manually verify the feature and "
        "target time windows."
    )

    print(
        "2. Confirm that no post-cutoff transaction "
        "data appears in the model features."
    )

    print(
        "3. Prefer chronological validation "
        "before final submission."
    )

else:

    print(
        "1. No critical statistical leakage "
        "was detected."
    )

    print(
        "2. Complete a manual temporal review."
    )

    print(
        "3. Continue with Optuna, MLflow, "
        "Evidently, and dashboard integration."
    )

# ============================================================
# Completion
# ============================================================

print("\n" + "=" * 70)

print(
    f"AUDIT OUTCOME: "
    f"{audit_outcome}"
)

print(
    f"OVERALL LEAKAGE RISK: "
    f"{overall_leakage_risk}"
)

print("=" * 70)

print(
    "\nF04 TARGET LEAKAGE AUDIT "
    "COMPLETED SUCCESSFULLY"
)

print("=" * 70)
