# Loan Default Prediction

Predicts whether a loan applicant is likely to default (`Default`: 1 = defaulted, 0 = repaid),
so risky applications can be flagged for closer review.

## Dataset

Kaggle loan-default data (`data/loandata.csv`): 255,347 rows, 18 columns, target `Default`
with ~11.61% defaults. Features cover borrower demographics, income/employment,
loan terms, credit history, and loan purpose. `LoanID` is an identifier and excluded from modeling.

## Approach

- Cleaning: no missing values or duplicates; kept `MonthsEmployed == 0` and distribution tails as valid.
- EDA: defaulters skew younger, with higher rates, larger loans relative to income, and lower employment stability.
- Feature engineering: `loan_to_income = LoanAmount / Income` (strongest single signal).
- Preprocessing: median/most-frequent imputation, `StandardScaler` for Logistic Regression,
  `OrdinalEncoder` for ordered `Education`, one-hot for nominal fields. Split 80/20 stratified;
  preprocessing fitted on train only.
- Models tested: Logistic Regression, Decision Tree, Random Forest, XGBoost (all in Pipelines).
- Imbalance: compared normal vs `class_weight="balanced"`; balanced LR won on recall/F1 at the same ROC-AUC.
- Final model: Logistic Regression (`class_weight="balanced"`, `C=1.0`, `max_iter=1000`),
  tuned with a small `GridSearchCV` over `C` on train/CV only.

## Results (held-out test)

- Accuracy: 0.6906
- Precision: 0.2280
- Recall: 0.6974
- F1: 0.3436
- ROC-AUC: 0.7616

Baseline (always predict repaid) gets ~88% accuracy but recall 0 — it catches no defaulters.
Threshold analysis on the final model: 0.3 → recall 0.91 / precision 0.16;
0.5 → recall 0.70 / precision 0.23; 0.6 → recall 0.55 / precision 0.28.
Lower thresholds catch more risk, higher ones raise fewer false alarms.

## Interpretability

Logistic Regression coefficients (on the standardized/one-hot features) show which inputs
push predictions toward default (positive) or repayment (negative). See the Phase 4 notebook
section for the top features chart. These are associations, not proof of causation.

## Running locally

1. Install requirements:
   `pip install -r requirements.txt`
2. Run the app:
   `streamlit run app.py`
