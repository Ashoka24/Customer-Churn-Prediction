# Customer Churn Analysis

An end-to-end Python project exploring customer churn, comparing two classification models, reporting held-out evaluation metrics, and providing SQL queries for descriptive segmentation.

> **Data note:** The included customer records are synthetic demonstration data. They do not represent a real company or customer population, and the metrics should not be presented as evidence about a real service provider.

## What is included

- Validation of required columns, customer IDs and churn labels
- Descriptive analysis by contract, tenure and service attributes
- Logistic regression and random forest classification pipelines
- Preprocessing fitted within each training pipeline
- Stratified train/test split and held-out precision, recall, F1 and ROC-AUC
- Confusion matrices, classification reports, feature rankings and ROC curves
- SQL summaries for contract type, tenure, support services and payment method
- Streamlit interface and automated tests

## Project structure

- `data/customer_churn.csv` — synthetic sample data
- `scripts/run_analysis.py` — data validation, modelling and evaluation
- `sql/churn_analysis.sql` — SQL case-study queries
- `app.py` — interactive descriptive interface
- `tests/` — automated tests
- `outputs/` — generated results, excluded from version control

## Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_analysis.py
python -m unittest discover -s tests -v
streamlit run app.py
```

## Interpreting results

Recall is useful when missing potential churners is costly; precision matters when outreach capacity is limited. ROC-AUC measures ranking across thresholds, but no single metric establishes business value. Compare models against simple baselines and choose thresholds based on the relative cost of false positives and false negatives.

Feature rankings are associations learned by a model, not causal explanations. Before real use, evaluate calibration, subgroup performance, data leakage, temporal stability and operational costs. A random holdout does not establish future-period performance.

## License

Code is provided under the MIT License. The synthetic dataset is for learning and portfolio demonstration only.
