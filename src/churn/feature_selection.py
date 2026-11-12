# ============================================================
# RetailPulse
# F4 Customer Churn Prediction
# XGBoost-Based Feature Selection
# ============================================================

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

INPUT_PATH = "data/processed/advanced_churn_features.csv"
OUTPUT_DATA_PATH = "data/processed/selected_churn_features.csv"

MODEL_PATH = "models/churn/feature_selection_model.pkl"

FEATURE_RANKING_PATH = "outputs/reports/feature_ranking.csv"
SELECTED_FEATURES_PATH = "outputs/reports/selected_features.txt"
SELECTION_REPORT_PATH = "outputs/reports/feature_selection_report.txt"

IMPORTANCE_PLOT_PATH = (
    "outputs/figures/selected_feature_importance.png"
)

TARGET = "Churn"
IDENTIFIER_COLUMNS = ["CustomerID"]

# Minimum importance required for automatic selection.
# Part 2 will also ensure that a minimum number of features is retained.
IMPORTANCE_THRESHOLD = 0.015

MINIMUM_FEATURES = 8
MAXIMUM_FEATURES = 15

# ============================================================
# Create Required Folders
# ============================================================

os.makedirs("data/processed", exist_ok=True)
os.makedirs("models/churn", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

print("=" * 60)
print("XGBOOST FEATURE SELECTION")
print("=" * 60)

# ============================================================
# Load Dataset
# ============================================================

print("\nLoading advanced churn dataset...")

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_PATH}\n"
        "Run advanced_feature_engineering.py first."
    )

df = pd.read_csv(INPUT_PATH)

print(f"Dataset Shape : {df.shape}")

# ============================================================
# Validate Dataset
# ============================================================

required_columns = [TARGET] + IDENTIFIER_COLUMNS

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing from dataset: {missing_columns}"
    )

if df.empty:
    raise ValueError("The churn dataset is empty.")

if df[TARGET].nunique() != 2:
    raise ValueError(
        f"Target column '{TARGET}' must contain exactly two classes."
    )

print("Dataset validation completed.")

# ============================================================
# Select Candidate Features
# ============================================================

excluded_columns = IDENTIFIER_COLUMNS + [TARGET]

candidate_features = [
    column
    for column in df.columns
    if column not in excluded_columns
]

non_numeric_features = [
    column
    for column in candidate_features
    if not pd.api.types.is_numeric_dtype(df[column])
]

if non_numeric_features:
    print("\nRemoving non-numeric features:")

    for column in non_numeric_features:
        print(f"- {column}")

    candidate_features = [
        column
        for column in candidate_features
        if column not in non_numeric_features
    ]

if not candidate_features:
    raise ValueError("No numeric candidate features were found.")

print(f"\nCandidate Features : {len(candidate_features)}")

# ============================================================
# Prepare Feature Matrix and Target
# ============================================================

X = df[candidate_features].copy()
y = df[TARGET].astype(int).copy()

# Replace invalid values before training.
X.replace([np.inf, -np.inf], np.nan, inplace=True)

missing_value_counts = X.isna().sum()

columns_with_missing_values = missing_value_counts[
    missing_value_counts > 0
]

if not columns_with_missing_values.empty:
    print("\nFilling missing values with feature medians...")

    for column in columns_with_missing_values.index:
        X[column] = X[column].fillna(X[column].median())

print("\nTarget Distribution")

target_distribution = y.value_counts().sort_index()

for class_value, count in target_distribution.items():
    percentage = count / len(y) * 100

    print(
        f"Class {class_value}: "
        f"{count} customers ({percentage:.2f}%)"
    )

# ============================================================
# Train/Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"\nTraining Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

# ============================================================
# Baseline Feature-Selection Model
# ============================================================

selection_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=300,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=1,
    gamma=0.1,
    subsample=0.7,
    colsample_bytree=1.0,
    reg_alpha=0.1,
    reg_lambda=1.5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

print("\nTraining feature-selection model...")

selection_model.fit(X_train, y_train)

print("Feature-selection model trained successfully.")

# ============================================================
# Baseline Model Evaluation
# ============================================================

baseline_prediction = selection_model.predict(X_test)

baseline_probability = selection_model.predict_proba(
    X_test
)[:, 1]

baseline_metrics = {
    "Accuracy": accuracy_score(
        y_test,
        baseline_prediction,
    ),
    "Precision": precision_score(
        y_test,
        baseline_prediction,
        zero_division=0,
    ),
    "Recall": recall_score(
        y_test,
        baseline_prediction,
        zero_division=0,
    ),
    "F1": f1_score(
        y_test,
        baseline_prediction,
        zero_division=0,
    ),
    "ROC-AUC": roc_auc_score(
        y_test,
        baseline_probability,
    ),
}

print("\n" + "=" * 60)
print("BASELINE PERFORMANCE — ALL FEATURES")
print("=" * 60)

for metric_name, metric_value in baseline_metrics.items():
    print(f"{metric_name:<12}: {metric_value:.4f}")

# ============================================================
# Calculate Feature Importance
# ============================================================

feature_ranking = pd.DataFrame(
    {
        "Feature": candidate_features,
        "Importance": selection_model.feature_importances_,
    }
)

feature_ranking.sort_values(
    by="Importance",
    ascending=False,
    inplace=True,
)

feature_ranking.reset_index(
    drop=True,
    inplace=True,
)

feature_ranking["Rank"] = (
    feature_ranking.index + 1
)

feature_ranking["CumulativeImportance"] = (
    feature_ranking["Importance"].cumsum()
)

print("\n" + "=" * 60)
print("TOP 15 FEATURES")
print("=" * 60)

for _, row in feature_ranking.head(15).iterrows():
    print(
        f"{int(row['Rank']):>2}. "
        f"{row['Feature']:<28} "
        f"{row['Importance']:.4f}"
    )

print("\nFeature importance ranking completed.")
print("Proceeding to automatic feature selection...")

# ============================================================
# Additional Imports for Cross-Validation
# ============================================================

from sklearn.base import clone

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)

# ============================================================
# Automatic Feature Selection
# ============================================================

# Initially select features whose importance is greater than
# or equal to the configured threshold.

threshold_features = feature_ranking.loc[
    feature_ranking["Importance"] >= IMPORTANCE_THRESHOLD,
    "Feature",
].tolist()

print("\nFeatures meeting importance threshold:")
print(len(threshold_features))

# Ensure that at least MINIMUM_FEATURES are retained.

if len(threshold_features) < MINIMUM_FEATURES:

    print(
        f"\nOnly {len(threshold_features)} features met the "
        f"threshold. Selecting the top {MINIMUM_FEATURES} features."
    )

    selected_features = feature_ranking.head(
        MINIMUM_FEATURES
    )["Feature"].tolist()

# Restrict the final feature set to MAXIMUM_FEATURES.

elif len(threshold_features) > MAXIMUM_FEATURES:

    print(
        f"\n{len(threshold_features)} features met the threshold. "
        f"Keeping only the top {MAXIMUM_FEATURES} features."
    )

    selected_features = feature_ranking.head(
        MAXIMUM_FEATURES
    )["Feature"].tolist()

else:

    selected_features = threshold_features

print("\n" + "=" * 60)
print("AUTOMATICALLY SELECTED FEATURES")
print("=" * 60)

for position, feature in enumerate(
    selected_features,
    start=1,
):
    importance_value = feature_ranking.loc[
        feature_ranking["Feature"] == feature,
        "Importance",
    ].iloc[0]

    print(
        f"{position:>2}. "
        f"{feature:<28} "
        f"{importance_value:.4f}"
    )

print(f"\nTotal Selected Features : {len(selected_features)}")

# ============================================================
# Prepare Selected Feature Matrices
# ============================================================

X_train_selected = X_train[selected_features].copy()

X_test_selected = X_test[selected_features].copy()

# ============================================================
# Cross-Validation Configuration
# ============================================================

cv_strategy = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

print("\nRunning 5-fold cross-validation...")

# ============================================================
# Full Feature-Set Cross-Validation
# ============================================================

full_feature_model = clone(selection_model)

full_cv_scores = cross_val_score(
    estimator=full_feature_model,
    X=X_train,
    y=y_train,
    scoring="roc_auc",
    cv=cv_strategy,
    n_jobs=-1,
)

full_cv_mean = full_cv_scores.mean()

full_cv_std = full_cv_scores.std()

print(
    "\nAll Features CV ROC-AUC "
    f": {full_cv_mean:.4f} ± {full_cv_std:.4f}"
)

# ============================================================
# Selected Feature-Set Cross-Validation
# ============================================================

selected_cv_model = clone(selection_model)

selected_cv_scores = cross_val_score(
    estimator=selected_cv_model,
    X=X_train_selected,
    y=y_train,
    scoring="roc_auc",
    cv=cv_strategy,
    n_jobs=-1,
)

selected_cv_mean = selected_cv_scores.mean()

selected_cv_std = selected_cv_scores.std()

print(
    "Selected Features CV ROC-AUC "
    f": {selected_cv_mean:.4f} ± {selected_cv_std:.4f}"
)

# ============================================================
# Validate Selected Feature Set
# ============================================================

# A reduced feature set is considered competitive when its
# cross-validation ROC-AUC is no more than 0.005 below the
# full feature-set cross-validation score.

CV_TOLERANCE = 0.005

selected_is_competitive = (
    selected_cv_mean
    >= full_cv_mean - CV_TOLERANCE
)

if selected_is_competitive:

    print(
        "\nSelected feature set is competitive."
    )

    final_features = selected_features

    selection_decision = (
        "Selected feature set retained because its "
        "cross-validation ROC-AUC was competitive."
    )

else:

    print(
        "\nSelected feature set lost too much ROC-AUC."
    )

    print(
        "Falling back to the top "
        f"{MAXIMUM_FEATURES} ranked features."
    )

    final_features = feature_ranking.head(
        MAXIMUM_FEATURES
    )["Feature"].tolist()

    selection_decision = (
        "Initial reduced feature set was not competitive. "
        f"The top {MAXIMUM_FEATURES} ranked features were retained."
    )

# ============================================================
# Final Selected Feature Matrices
# ============================================================

X_train_final = X_train[final_features].copy()

X_test_final = X_test[final_features].copy()

print("\n" + "=" * 60)
print("FINAL FEATURE SET")
print("=" * 60)

for position, feature in enumerate(
    final_features,
    start=1,
):
    print(f"{position:>2}. {feature}")

print(f"\nFinal Feature Count : {len(final_features)}")

# ============================================================
# Train Reduced XGBoost Model
# ============================================================

selected_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=300,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=1,
    gamma=0.1,
    subsample=0.7,
    colsample_bytree=1.0,
    reg_alpha=0.1,
    reg_lambda=1.5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

print("\nTraining model with final selected features...")

selected_model.fit(
    X_train_final,
    y_train,
)

print("Selected-feature model trained successfully.")

# ============================================================
# Selected Model Predictions
# ============================================================

selected_prediction = selected_model.predict(
    X_test_final
)

selected_probability = selected_model.predict_proba(
    X_test_final
)[:, 1]

# ============================================================
# Selected Model Metrics
# ============================================================

selected_metrics = {
    "Accuracy": accuracy_score(
        y_test,
        selected_prediction,
    ),
    "Precision": precision_score(
        y_test,
        selected_prediction,
        zero_division=0,
    ),
    "Recall": recall_score(
        y_test,
        selected_prediction,
        zero_division=0,
    ),
    "F1": f1_score(
        y_test,
        selected_prediction,
        zero_division=0,
    ),
    "ROC-AUC": roc_auc_score(
        y_test,
        selected_probability,
    ),
}

print("\n" + "=" * 60)
print("SELECTED-FEATURE MODEL PERFORMANCE")
print("=" * 60)

for metric_name, metric_value in selected_metrics.items():
    print(f"{metric_name:<12}: {metric_value:.4f}")

# ============================================================
# Compare Full and Selected Models
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

comparison_rows = []

for metric_name in baseline_metrics:

    baseline_value = baseline_metrics[metric_name]

    selected_value = selected_metrics[metric_name]

    difference = selected_value - baseline_value

    comparison_rows.append(
        {
            "Metric": metric_name,
            "AllFeatures": baseline_value,
            "SelectedFeatures": selected_value,
            "Difference": difference,
        }
    )

    print(
        f"{metric_name:<12} "
        f"All: {baseline_value:.4f} | "
        f"Selected: {selected_value:.4f} | "
        f"Difference: {difference:+.4f}"
    )

comparison_df = pd.DataFrame(comparison_rows)

# ============================================================
# Determine Recommended Model
# ============================================================

if selected_metrics["ROC-AUC"] >= baseline_metrics["ROC-AUC"]:

    recommended_model_name = "Selected-Feature Model"

    recommended_model = selected_model

    recommended_features = final_features

    recommended_metrics = selected_metrics

else:

    recommended_model_name = "All-Feature Model"

    recommended_model = selection_model

    recommended_features = candidate_features

    recommended_metrics = baseline_metrics

print("\nRecommended Model")
print("------------------------------")
print(recommended_model_name)

print(
    f"Recommended ROC-AUC : "
    f"{recommended_metrics['ROC-AUC']:.4f}"
)

print("\nProceeding to save selected data, models, and reports...")

# ============================================================
# Save Feature Ranking
# ============================================================

feature_ranking.to_csv(
    FEATURE_RANKING_PATH,
    index=False,
)

print(f"\nFeature ranking saved: {FEATURE_RANKING_PATH}")

# ============================================================
# Save Model Comparison
# ============================================================

comparison_path = (
    "outputs/reports/feature_selection_comparison.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False,
)

print(f"Model comparison saved: {comparison_path}")

# ============================================================
# Create Recommended Feature Dataset
# ============================================================

# The saved dataset must contain:
# - Customer identifier
# - Recommended model features
# - Churn target

output_columns = (
    IDENTIFIER_COLUMNS
    + recommended_features
    + [TARGET]
)

selected_dataset = df[output_columns].copy()

selected_dataset.to_csv(
    OUTPUT_DATA_PATH,
    index=False,
)

print(
    f"Selected feature dataset saved: "
    f"{OUTPUT_DATA_PATH}"
)

print(
    f"Selected dataset shape: "
    f"{selected_dataset.shape}"
)

# ============================================================
# Save Recommended Features
# ============================================================

with open(
    SELECTED_FEATURES_PATH,
    "w",
    encoding="utf-8",
) as feature_file:

    feature_file.write(
        "RetailPulse Selected Churn Features\n"
    )

    feature_file.write("=" * 60)
    feature_file.write("\n\n")

    feature_file.write(
        f"Recommended Model: "
        f"{recommended_model_name}\n"
    )

    feature_file.write(
        f"Number of Features: "
        f"{len(recommended_features)}\n\n"
    )

    feature_file.write("Features\n")
    feature_file.write("-" * 60)
    feature_file.write("\n")

    for position, feature in enumerate(
        recommended_features,
        start=1,
    ):
        feature_file.write(
            f"{position}. {feature}\n"
        )

print(
    f"Selected features saved: "
    f"{SELECTED_FEATURES_PATH}"
)

# ============================================================
# Save Recommended Model
# ============================================================

joblib.dump(
    recommended_model,
    MODEL_PATH,
)

print(
    f"Recommended feature-selection model saved: "
    f"{MODEL_PATH}"
)

# ============================================================
# Save Recommended Model Feature Order
# ============================================================

feature_order_path = (
    "models/churn/feature_selection_features.pkl"
)

joblib.dump(
    recommended_features,
    feature_order_path,
)

print(
    f"Model feature order saved: "
    f"{feature_order_path}"
)

# ============================================================
# Feature Importance Plot
# ============================================================

plot_data = feature_ranking.head(20).copy()

plot_data = plot_data.sort_values(
    by="Importance",
    ascending=True,
)

plt.figure(figsize=(10, 8))

plt.barh(
    plot_data["Feature"],
    plot_data["Importance"],
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")

plt.title(
    "Top 20 XGBoost Feature Importances"
)

plt.tight_layout()

plt.savefig(
    IMPORTANCE_PLOT_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    f"Feature importance plot saved: "
    f"{IMPORTANCE_PLOT_PATH}"
)

# ============================================================
# Save Feature-Selection Report
# ============================================================

with open(
    SELECTION_REPORT_PATH,
    "w",
    encoding="utf-8",
) as report:

    report.write("RetailPulse\n")
    report.write("Customer Churn Feature Selection Report\n")
    report.write("=" * 70)
    report.write("\n\n")

    # Dataset information

    report.write("Dataset Information\n")
    report.write("-" * 70)
    report.write("\n")

    report.write(
        f"Input Dataset: {INPUT_PATH}\n"
    )

    report.write(
        f"Original Dataset Shape: {df.shape}\n"
    )

    report.write(
        f"Candidate Feature Count: "
        f"{len(candidate_features)}\n"
    )

    report.write(
        f"Training Samples: {len(X_train)}\n"
    )

    report.write(
        f"Testing Samples: {len(X_test)}\n\n"
    )

    # Selection configuration

    report.write("Selection Configuration\n")
    report.write("-" * 70)
    report.write("\n")

    report.write(
        f"Importance Threshold: "
        f"{IMPORTANCE_THRESHOLD}\n"
    )

    report.write(
        f"Minimum Features: "
        f"{MINIMUM_FEATURES}\n"
    )

    report.write(
        f"Maximum Features: "
        f"{MAXIMUM_FEATURES}\n"
    )

    report.write(
        f"Cross-Validation Folds: 5\n"
    )

    report.write(
        f"CV Tolerance: {CV_TOLERANCE}\n\n"
    )

    # Baseline metrics

    report.write(
        "All-Feature Model Test Metrics\n"
    )

    report.write("-" * 70)
    report.write("\n")

    for metric_name, metric_value in (
        baseline_metrics.items()
    ):
        report.write(
            f"{metric_name}: "
            f"{metric_value:.4f}\n"
        )

    report.write("\n")

    # Selected model metrics

    report.write(
        "Selected-Feature Model Test Metrics\n"
    )

    report.write("-" * 70)
    report.write("\n")

    for metric_name, metric_value in (
        selected_metrics.items()
    ):
        report.write(
            f"{metric_name}: "
            f"{metric_value:.4f}\n"
        )

    report.write("\n")

    # Cross-validation results

    report.write("Cross-Validation Results\n")
    report.write("-" * 70)
    report.write("\n")

    report.write(
        "All Features CV ROC-AUC: "
        f"{full_cv_mean:.4f} "
        f"+/- {full_cv_std:.4f}\n"
    )

    report.write(
        "Selected Features CV ROC-AUC: "
        f"{selected_cv_mean:.4f} "
        f"+/- {selected_cv_std:.4f}\n\n"
    )

    # Selection result

    report.write("Selection Decision\n")
    report.write("-" * 70)
    report.write("\n")

    report.write(
        f"{selection_decision}\n\n"
    )

    report.write(
        f"Recommended Model: "
        f"{recommended_model_name}\n"
    )

    report.write(
        f"Recommended ROC-AUC: "
        f"{recommended_metrics['ROC-AUC']:.4f}\n"
    )

    report.write(
        f"Recommended Feature Count: "
        f"{len(recommended_features)}\n\n"
    )

    # Recommended features

    report.write("Recommended Features\n")
    report.write("-" * 70)
    report.write("\n")

    for position, feature in enumerate(
        recommended_features,
        start=1,
    ):
        importance_match = feature_ranking.loc[
            feature_ranking["Feature"] == feature,
            "Importance",
        ]

        if not importance_match.empty:
            importance_value = (
                importance_match.iloc[0]
            )
        else:
            importance_value = 0.0

        report.write(
            f"{position:>2}. "
            f"{feature:<30} "
            f"{importance_value:.6f}\n"
        )

    report.write("\n")

    # Complete feature ranking

    report.write("Complete Feature Ranking\n")
    report.write("-" * 70)
    report.write("\n")

    report.write(
        feature_ranking[
            [
                "Rank",
                "Feature",
                "Importance",
                "CumulativeImportance",
            ]
        ].to_string(index=False)
    )

    report.write("\n")

print(
    f"Feature-selection report saved: "
    f"{SELECTION_REPORT_PATH}"
)

# ============================================================
# Console Feature Ranking
# ============================================================

print("\n" + "=" * 60)
print("RECOMMENDED FEATURES")
print("=" * 60)

for position, feature in enumerate(
    recommended_features,
    start=1,
):
    importance_value = feature_ranking.loc[
        feature_ranking["Feature"] == feature,
        "Importance",
    ].iloc[0]

    print(
        f"{position:>2}. "
        f"{feature:<30} "
        f"{importance_value:.4f}"
    )

# ============================================================
# Final Performance Summary
# ============================================================

print("\n" + "=" * 60)
print("FEATURE-SELECTION SUMMARY")
print("=" * 60)

print(
    f"Candidate Features       : "
    f"{len(candidate_features)}"
)

print(
    f"Automatically Selected   : "
    f"{len(final_features)}"
)

print(
    f"Recommended Features     : "
    f"{len(recommended_features)}"
)

print(
    f"Recommended Model        : "
    f"{recommended_model_name}"
)

print(
    f"All Features ROC-AUC     : "
    f"{baseline_metrics['ROC-AUC']:.4f}"
)

print(
    f"Selected Features ROC-AUC: "
    f"{selected_metrics['ROC-AUC']:.4f}"
)

print(
    f"Recommended ROC-AUC      : "
    f"{recommended_metrics['ROC-AUC']:.4f}"
)

# ============================================================
# Generated Files
# ============================================================

generated_files = [
    OUTPUT_DATA_PATH,
    MODEL_PATH,
    feature_order_path,
    FEATURE_RANKING_PATH,
    SELECTED_FEATURES_PATH,
    SELECTION_REPORT_PATH,
    comparison_path,
    IMPORTANCE_PLOT_PATH,
]

print("\nGenerated Files")
print("-" * 60)

for generated_file in generated_files:
    print(f"✔ {generated_file}")

# ============================================================
# Final Recommendation
# ============================================================

print("\n" + "=" * 60)

roc_improvement = (
    selected_metrics["ROC-AUC"]
    - baseline_metrics["ROC-AUC"]
)

if recommended_model_name == (
    "Selected-Feature Model"
):

    print(
        "The reduced feature set produced the "
        "best test ROC-AUC."
    )

    print(
        "Use selected_churn_features.csv for "
        "the next training stage."
    )

elif roc_improvement >= -CV_TOLERANCE:

    print(
        "The selected model was competitive, "
        "but the all-feature model had a slightly "
        "higher test ROC-AUC."
    )

else:

    print(
        "The reduced feature set did not improve "
        "the test ROC-AUC."
    )

    print(
        "The recommended dataset therefore retains "
        "the stronger feature configuration."
    )

print("=" * 60)

# ============================================================
# Finished
# ============================================================

print(
    "\nFEATURE SELECTION COMPLETED SUCCESSFULLY"
)

print("=" * 60)