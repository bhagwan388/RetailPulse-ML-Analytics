# RetailPulse — Demand Forecasting & Customer Churn Prediction

**Portfolio extract from a collaborative internship project (July 2026).** This repository isolates my assigned **F03 Demand Forecasting** and **F04 Customer Churn Prediction** modules. Other team members' F01, F02, F05, F06 implementations are intentionally excluded.

## Scope and technical work

### F03 — Demand Forecasting
- Aggregated transactions into a 739-day daily revenue series.
- Implemented Prophet and PyTorch LSTM forecasting pipelines and a hybrid experiment.
- Produced a 30-day revenue forecast and export for downstream inventory integration.
- **Reported evaluation:** Prophet MAPE 25.00%; LSTM MAPE 46.95%. The 12% target was not met. Hybrid evaluation requires correction; do not claim zero hybrid error.
- Forecasts are **daily revenue**, used as a proxy for demand, **not SKU-level unit demand**.

### F04 — Customer Churn Prediction
- Built time-cutoff customer churn labels and recency/frequency/monetary-based features.
- Trained, tuned, and compared XGBoost models; added feature selection, leakage audit, and SHAP explanations.
- **Reported held-out results (six-feature model):** ROC-AUC 0.8187, recall 0.9097, Precision@Top20% 0.9009 (191 actual churners among 212 highest-risk customers). The 0.88 ROC-AUC target was not met.
- The original report recommends further temporal validation.

## Layout

- `src/forecasting/` — original non-empty F03 Python scripts
- `src/churn/` — original non-empty F04 Python scripts
- `requirements.txt` — initial dependency list (not yet reproducibility-tested)

## Important setup note

This is a **source-only curated portfolio extract**, not yet a turnkey application. The scripts preserve their original relative paths and may expect data files, saved models, and output folders from the full team repository. To run the complete pipelines, review each script's expected paths, supply licensed or synthetic datasets, and verify dependencies. No performance numbers have been re-evaluated in this extract.

Do not upload private customer records or sensitive datasets. Obtain permission before publishing any shared internship material.

## Next steps

1. Refactor paths and configuration to run independently.
2. Add a small synthetic dataset and smoke tests.
3. Reproduce evaluation metrics and document exact environment versions.
4. Add a standalone Streamlit demonstration and screenshots.

## Attribution

Collaborative RetailPulse internship project. This repository documents only the F03 and F04 assigned contributions. Other modules and the original team dashboard are not claimed as individual work.
