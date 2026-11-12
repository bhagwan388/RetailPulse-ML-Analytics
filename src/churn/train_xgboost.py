# ============================================================
# RetailPulse
# F4 Customer Churn Prediction
# Train Optimized XGBoost Model
# ============================================================

import os
import joblib
import warnings

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
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

warnings.filterwarnings("ignore")

# ============================================================
# Create Folders
# ============================================================

os.makedirs("models/churn", exist_ok=True)

os.makedirs("outputs/reports", exist_ok=True)

os.makedirs("outputs/figures", exist_ok=True)

print("=" * 60)
print("CUSTOMER CHURN PREDICTION")
print("=" * 60)

# ============================================================
# Load Dataset
# ============================================================

print("\nLoading Advanced Feature Dataset...")

df = pd.read_csv(
    "data/processed/advanced_churn_features.csv"
)

print(f"Dataset Shape : {df.shape}")

# ============================================================
# Feature Selection
# ============================================================

TARGET = "Churn"

FEATURES = [

    column

    for column in df.columns

    if column not in [

        "CustomerID",

        TARGET,

    ]

]

print(f"\nTotal Features : {len(FEATURES)}")

print("\nFeature List")

for feature in FEATURES:

    print(f"✔ {feature}")

# ============================================================
# Split Dataset
# ============================================================

X = df[FEATURES]

y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y,

)

print("\nTraining Samples :", len(X_train))

print("Testing Samples  :", len(X_test))

# ============================================================
# Train Model
# ============================================================

print("\nTraining Optimized XGBoost...")

model = XGBClassifier(

    objective="binary:logistic",

    eval_metric="logloss",

    random_state=42,

    n_estimators=300,

    learning_rate=0.05,

    max_depth=6,

    subsample=0.8,

    colsample_bytree=0.8,

)

model.fit(

    X_train,

    y_train,

)

print("Training Completed.")

# ============================================================
# Prediction
# ============================================================

prediction = model.predict(X_test)

probability = model.predict_proba(

    X_test

)[:, 1]

# ============================================================
# Evaluation Metrics
# ============================================================

accuracy = accuracy_score(

    y_test,

    prediction,

)

precision = precision_score(

    y_test,

    prediction,

)

recall = recall_score(

    y_test,

    prediction,

)

f1 = f1_score(

    y_test,

    prediction,

)

roc_auc = roc_auc_score(

    y_test,

    probability,

)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nClassification Report\n")

print(

    classification_report(

        y_test,

        prediction,

    )

)

print("\nProceeding to save model and generate reports...")

# ============================================================
# Save Model
# ============================================================

model_path = "models/churn/best_xgboost.pkl"

joblib.dump(
    model,
    model_path,
)

print("\nBest model saved successfully.")

# ============================================================
# Save Metrics Report
# ============================================================

report_path = "outputs/reports/churn_metrics.txt"

with open(report_path, "w") as report:

    report.write("RetailPulse\n")
    report.write("Customer Churn Prediction\n")
    report.write("=" * 60)
    report.write("\n\n")

    report.write(f"Dataset Shape : {df.shape}\n")
    report.write(f"Training Samples : {len(X_train)}\n")
    report.write(f"Testing Samples : {len(X_test)}\n\n")

    report.write("Performance Metrics\n")
    report.write("----------------------------\n")

    report.write(f"Accuracy : {accuracy:.4f}\n")
    report.write(f"Precision : {precision:.4f}\n")
    report.write(f"Recall : {recall:.4f}\n")
    report.write(f"F1 Score : {f1:.4f}\n")
    report.write(f"ROC-AUC : {roc_auc:.4f}\n")

print("Metrics report saved.")

# ============================================================
# Save Classification Report
# ============================================================

classification_path = (
    "outputs/reports/classification_report.txt"
)

with open(classification_path, "w") as report:

    report.write(

        classification_report(
            y_test,
            prediction,
        )

    )

print("Classification report saved.")

# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    prediction,
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "outputs/figures/confusion_matrix.png",
    dpi=300,
)

plt.close()

print("Confusion Matrix saved.")

# ============================================================
# ROC Curve
# ============================================================

RocCurveDisplay.from_predictions(
    y_test,
    probability,
)

plt.title("ROC Curve")

plt.tight_layout()

plt.savefig(
    "outputs/figures/roc_curve.png",
    dpi=300,
)

plt.close()

print("ROC Curve saved.")

# ============================================================
# Feature Importance
# ============================================================

importance = pd.DataFrame(

    {

        "Feature": FEATURES,

        "Importance": model.feature_importances_,

    }

)

importance = importance.sort_values(

    by="Importance",

    ascending=False,

)

plt.figure(figsize=(10, 8))

plt.barh(

    importance["Feature"],

    importance["Importance"],

)

plt.gca().invert_yaxis()

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.title("XGBoost Feature Importance")

plt.tight_layout()

plt.savefig(

    "outputs/figures/feature_importance.png",

    dpi=300,

)

plt.close()

print("Feature importance plot saved.")

# ============================================================
# Save Feature Importance CSV
# ============================================================

importance.to_csv(

    "outputs/reports/feature_importance.csv",

    index=False,

)

print("Feature importance CSV saved.")

print("\nGenerating final report...")

# ============================================================
# Save Prediction Results
# ============================================================

prediction_results = X_test.copy()

prediction_results["ActualChurn"] = y_test.values

prediction_results["PredictedChurn"] = prediction

prediction_results["ChurnProbability"] = probability

prediction_results = prediction_results.sort_values(
    by="ChurnProbability",
    ascending=False,
)

prediction_results.to_csv(
    "outputs/reports/churn_predictions.csv",
    index=False,
)

print("Prediction CSV saved.")

# ============================================================
# Top High-Risk Customers
# ============================================================

top_customers = prediction_results.head(20)

top_customers.to_csv(
    "outputs/reports/top_20_churn_customers.csv",
    index=False,
)

print("Top 20 high-risk customers saved.")

# ============================================================
# Display Top Important Features
# ============================================================

print("\n" + "=" * 60)
print("TOP FEATURE IMPORTANCE")
print("=" * 60)

for i, row in importance.head(10).iterrows():

    print(
        f"{row['Feature']:<30} "
        f"{row['Importance']:.4f}"
    )

# ============================================================
# Overall Summary
# ============================================================

print("\n" + "=" * 60)
print("MODEL SUMMARY")
print("=" * 60)

print(f"Dataset Shape      : {df.shape}")
print(f"Total Features     : {len(FEATURES)}")
print(f"Training Samples   : {len(X_train)}")
print(f"Testing Samples    : {len(X_test)}")

print("\nPerformance")

print(f"Accuracy           : {accuracy:.4f}")
print(f"Precision          : {precision:.4f}")
print(f"Recall             : {recall:.4f}")
print(f"F1 Score           : {f1:.4f}")
print(f"ROC-AUC            : {roc_auc:.4f}")

# ============================================================
# Generated Files
# ============================================================

print("\nGenerated Files")
print("-" * 60)

generated_files = [

    "models/churn/best_xgboost.pkl",

    "outputs/reports/churn_metrics.txt",

    "outputs/reports/classification_report.txt",

    "outputs/reports/feature_importance.csv",

    "outputs/reports/churn_predictions.csv",

    "outputs/reports/top_20_churn_customers.csv",

    "outputs/figures/confusion_matrix.png",

    "outputs/figures/roc_curve.png",

    "outputs/figures/feature_importance.png",

]

for file in generated_files:

    print(f"✔ {file}")

# ============================================================
# Recommendation
# ============================================================

print("\n" + "=" * 60)

if roc_auc >= 0.88:

    print("Excellent! Target ROC-AUC achieved.")

elif roc_auc >= 0.85:

    print("Very good model.")
    print("Small improvements may achieve ROC-AUC ≥ 0.88.")

elif roc_auc >= 0.82:

    print("Good model.")
    print("Run hyperparameter_tuning.py to improve performance.")

else:

    print("Model performance is moderate.")
    print("Consider additional feature engineering and tuning.")

print("=" * 60)

# ============================================================
# Finished
# ============================================================

print("\nCUSTOMER CHURN PREDICTION COMPLETED SUCCESSFULLY")
print("=" * 60)
