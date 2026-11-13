# RetailPulse — ML Analytics & Intelligence Hub

An end-to-end machine learning portfolio project demonstrating
retail revenue forecasting, customer churn prediction, temporal
evaluation, and interactive analytics.

## Project overview

RetailPulse combines two independently developed modules:

- F03: Daily revenue forecasting
- F04: Customer churn prediction

This repository is a curated extract from a collaborative
internship project (July 2026). It documents my assigned
F03 and F04 contributions. Other team members' modules
are not claimed as my individual work.

## Interactive dashboard

The Streamlit public demo contains five tabs:

1. Synthetic customer analytics overview
2. Historical demand forecasting
3. Synthetic customer risk explorer
4. Historical model evaluation
5. Simulated feature contributions

Run locally:

    pip install -r dashboard/requirements.txt
    streamlit run dashboard/public_app.py

## F03 — Revenue forecasting

- Daily transaction series: 739 days
- Model: Facebook Prophet
- Forecast horizon: 30 days
- Evaluation: Five retrospective 30-day test windows
- Baseline: Seasonal naive, repeating the previous week's values

Average rolling-backtest results:

| Metric | Prophet | Seasonal naive |
|---|---:|---:|
| MAE | 10,193.72 | 12,232.29 |
| RMSE | 15,857.58 | 17,627.94 |
| WAPE | 28.21% | 34.67% |

Prophet achieved lower MAE in four of five windows.
The average window MAE was approximately 16.7% lower
than the baseline.

These are retrospective results on historical data,
not production performance guarantees.

The forecasts represent aggregate daily revenue,
not SKU-level unit demand. Currency is not independently verified.

## F04 — Customer churn prediction

Model: XGBoost

Evaluation uses chronological training, validation,
and an out-of-time test snapshot.

Historical final test:

| Metric | Result |
|---|---:|
| Test customers | 5,281 |
| ROC-AUC | 0.7949 |
| Accuracy | 71.79% |
| Precision | 68.66% |
| Recall | 92.34% |
| F1 | 0.7876 |
| Precision at top 20% | 88.17% |

The churn prediction horizon is 90 days.

These results are historical evaluation metrics.
They do not describe the synthetic customers in the public demo.

## Public demonstration data

The public dashboard uses synthetic customer IDs,
artificial churn outcomes, simulated risk scores,
and randomly generated feature contributions.

The simulated contributions are NOT genuine SHAP
explanations and are not linked to the risk scores.

Historical aggregate evaluation reports and forecasting
artifacts are included for research demonstration,
subject to applicable publication permissions.

The original transaction dataset, trained models,
and customer-level test records are excluded.

## Technology

Python, Pandas, NumPy, Prophet, XGBoost,
scikit-learn, SHAP, Plotly, and Streamlit.

## Repository structure

- dashboard/app.py: Original local dashboard
- dashboard/public_app.py: Public demonstration
- dashboard/requirements.txt: Dashboard dependencies
- demo_data/: Public demonstration artifacts
- scripts/create_demo_data.py: Synthetic data generator
- src/forecasting/: F03 implementation
- src/churn/: F04 implementation

## Limitations

- Historical data is not a live retail feed.
- Prophet's yearly seasonality is weakly identified in
  early rolling windows with less than two years of history.
- The public dashboard is not a production inference service.
- Synthetic customer scores and simulated feature
  contributions must not be interpreted as model outputs.
- The historical evaluation reports do not validate
  the synthetic customer records.

## Attribution

RetailPulse was developed as a collaborative internship
project. This repository focuses on my assigned forecasting
and churn-prediction modules.
