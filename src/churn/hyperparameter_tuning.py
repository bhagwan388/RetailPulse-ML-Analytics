# ============================================================
# RetailPulse
# F4 Customer Churn Prediction
# Hyperparameter Tuning using RandomizedSearchCV
# ============================================================

import os
import json
import joblib
import warnings

import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

warnings.filterwarnings("ignore")

# ============================================================
# Create Output Folders
# ============================================================

os.makedirs("models/churn", exist_ok=True)

os.makedirs("outputs/reports", exist_ok=True)

os.makedirs("outputs/figures", exist_ok=True)

print("=" * 60)
print("HYPERPARAMETER TUNING - CUSTOMER CHURN")
print("=" * 60)

# ============================================================
# Load Dataset
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    "data/processed/selected_churn_features.csv"
)

print(f"Dataset Shape : {df.shape}")

# ============================================================
# Features
# ============================================================

TARGET = "Churn"

FEATURES = [
    column
    for column in df.columns
    if column not in ["CustomerID", TARGET]
]

print(f"Selected Feature Count : {len(FEATURES)}")

for feature in FEATURES:
    print(f"✔ {feature}")

TARGET = "Churn"

X = df[FEATURES]

y = df[TARGET]

# ============================================================
# Train Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

# ============================================================
# Base Model
# ============================================================

xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
)

# ============================================================
# Hyperparameter Search Space
# ============================================================

param_distributions = {

    "n_estimators": [
        100,
        200,
        300,
        400,
        500,
        600,
    ],

    "learning_rate": [
        0.01,
        0.03,
        0.05,
        0.1,
        0.2,
    ],

    "max_depth": [
        3,
        4,
        5,
        6,
        7,
        8,
        10,
    ],

    "min_child_weight": [
        1,
        3,
        5,
        7,
    ],

    "gamma": [
        0,
        0.1,
        0.2,
        0.3,
        0.5,
    ],

    "subsample": [
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ],

    "colsample_bytree": [
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ],

    "reg_alpha": [
        0,
        0.01,
        0.1,
        1,
    ],

    "reg_lambda": [
        1,
        1.5,
        2,
        3,
        5,
    ],

    "scale_pos_weight": [
        1,
        1.2,
        1.5,
        2,
        3,
    ],
}

# ============================================================
# Stratified Cross Validation
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

# ============================================================
# Randomized Search
# ============================================================

print("\nStarting RandomizedSearchCV...")
print("This may take several minutes...\n")

random_search = RandomizedSearchCV(

    estimator=xgb_model,

    param_distributions=param_distributions,

    n_iter=50,

    scoring="roc_auc",

    cv=cv,

    verbose=2,

    random_state=42,

    n_jobs=-1,

    return_train_score=True,

)

# ============================================================
# Model Training
# ============================================================

random_search.fit(X_train, y_train)

print("\nHyperparameter tuning completed.")

print("\nBest Cross Validation ROC-AUC")

print(f"{random_search.best_score_:.4f}")

print("\nBest Parameters")

for key, value in random_search.best_params_.items():
    print(f"{key:20}: {value}")

# ============================================================
# Best Model
# ============================================================

best_model = random_search.best_estimator_

prediction = best_model.predict(X_test)

probability = best_model.predict_proba(X_test)[:, 1]

print("\nEvaluating best model on test data...")

# ============================================================
# Evaluation Metrics
# ============================================================

accuracy = accuracy_score(y_test, prediction)

precision = precision_score(y_test, prediction)

recall = recall_score(y_test, prediction)

f1 = f1_score(y_test, prediction)

roc_auc = roc_auc_score(y_test, probability)

print("\n" + "=" * 60)
print("TEST SET PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

# ============================================================
# Classification Report
# ============================================================

class_report = classification_report(
    y_test,
    prediction
)

print("\nClassification Report\n")
print(class_report)

# ============================================================
# Save Best Model
# ============================================================

model_path = "models/churn/tuned_selected_xgboost.pkl"

joblib.dump(
    best_model,
    model_path
)

print(f"\nBest model saved to : {model_path}")

# ============================================================
# Save Best Parameters
# ============================================================

parameter_path = (
    "outputs/reports/selected_best_parameters.json"
)

with open(parameter_path, "w") as f:

    json.dump(
        random_search.best_params_,
        f,
        indent=4
    )

print("Best parameters saved.")

# ============================================================
# Save Complete Tuning Report
# ============================================================

report_path = (
    "outputs/reports/selected_tuning_report.txt"
)

with open(report_path, "w") as f:

    f.write("RetailPulse\n")
    f.write("Customer Churn Prediction\n")
    f.write("Hyperparameter Tuning Report\n")
    f.write("=" * 60)
    f.write("\n\n")

    f.write("Dataset Information\n")
    f.write("-------------------------\n")
    f.write(f"Total Samples : {len(df)}\n")
    f.write(f"Training Samples : {len(X_train)}\n")
    f.write(f"Testing Samples : {len(X_test)}\n\n")

    f.write("Selected Features\n")
    f.write("-------------------------\n")

    for feature in FEATURES:
        f.write(f"- {feature}\n")

    f.write("\n")

    f.write("Best Cross Validation ROC-AUC\n")
    f.write("-------------------------\n")
    f.write(f"{random_search.best_score_:.4f}\n\n")

    f.write("Best Parameters\n")
    f.write("-------------------------\n")

    for key, value in random_search.best_params_.items():
        f.write(f"{key}: {value}\n")

    f.write("\n")

    f.write("Test Metrics\n")
    f.write("-------------------------\n")

    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision : {precision:.4f}\n")
    f.write(f"Recall : {recall:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n")
    f.write(f"ROC-AUC : {roc_auc:.4f}\n\n")

    f.write("Classification Report\n")
    f.write("-------------------------\n")

    f.write(class_report)

print("Tuning report saved.")

# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    prediction
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "outputs/figures/confusion_matrix.png",
    dpi=300
)

plt.close()

print("Confusion Matrix saved.")

# ============================================================
# ROC Curve
# ============================================================

RocCurveDisplay.from_predictions(
    y_test,
    probability
)

plt.title("ROC Curve")

plt.tight_layout()

plt.savefig(
    "outputs/figures/roc_curve.png",
    dpi=300
)

plt.close()

print("ROC Curve saved.")

# ============================================================
# Feature Importance
# ============================================================

importance = pd.DataFrame(
    {
        "Feature": FEATURES,
        "Importance": best_model.feature_importances_,
    }
)

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(8,5))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.title("XGBoost Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "outputs/figures/feature_importance.png",
    dpi=300
)

plt.close()

print("Feature Importance plot saved.")

importance.to_csv(
    "outputs/reports/feature_importance.csv",
    index=False
)

print("Feature Importance CSV saved.")
# ============================================================
# Save Cross Validation Results
# ============================================================

cv_results = pd.DataFrame(random_search.cv_results_)

cv_results = cv_results.sort_values(
    by="rank_test_score"
)

cv_results.to_csv(
    "outputs/reports/cv_results.csv",
    index=False
)

print("Cross-validation results saved.")

# ============================================================
# Top 10 Hyperparameter Combinations
# ============================================================

print("\n" + "=" * 60)
print("TOP 10 HYPERPARAMETER COMBINATIONS")
print("=" * 60)

columns = [
    "rank_test_score",
    "mean_test_score",
    "std_test_score",
    "mean_train_score",
]

summary = cv_results[columns].head(10)

print(summary)

# ============================================================
# Best Feature Importance (Console)
# ============================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

for feature, score in zip(
    importance["Feature"],
    importance["Importance"],
):
    print(f"{feature:<20} {score:.4f}")

# ============================================================
# Improvement Summary
# ============================================================

print("\n" + "=" * 60)
print("MODEL SUMMARY")
print("=" * 60)

print(f"Cross Validation ROC-AUC : {random_search.best_score_:.4f}")
print(f"Test ROC-AUC             : {roc_auc:.4f}")
print(f"Accuracy                 : {accuracy:.4f}")
print(f"Precision                : {precision:.4f}")
print(f"Recall                   : {recall:.4f}")
print(f"F1 Score                 : {f1:.4f}")

# ============================================================
# Generated Files
# ============================================================

print("\nGenerated Files")
print("-" * 60)

generated_files = [
    "models/churn/tuned_selected_xgboost.pkl",
    "outputs/reports/selected_best_parameters.json",
    "outputs/reports/selected_tuning_report.txt",
    "outputs/reports/feature_importance.csv",
    "outputs/reports/cv_results.csv",
    "outputs/figures/confusion_matrix.png",
    "outputs/figures/roc_curve.png",
    "outputs/figures/feature_importance.png",
]

for file in generated_files:
    print(f"✔ {file}")

# ============================================================
# Finished
# ============================================================

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"\nBest Cross Validation ROC-AUC : {random_search.best_score_:.4f}")
print(f"Final Test ROC-AUC            : {roc_auc:.4f}")

if roc_auc >= 0.88:
    print("\nTarget Achieved (ROC-AUC ≥ 0.88)")
elif roc_auc >= 0.85:
    print("\nExcellent model. Minor feature engineering can push ROC-AUC above 0.88.")
elif roc_auc >= 0.82:
    print("\nGood improvement. Add more customer behavior features to increase ROC-AUC.")
else:
    print("\nModel needs additional feature engineering before reaching the target.")

print("\nRetailPulse - F4 Customer Churn Hyperparameter Tuning Finished.")
