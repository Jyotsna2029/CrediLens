# Multi-model Credit Risk Assessment with SHAP

Extension of https://github.com/tanishq-latent/Credit-Risk-Assesment-using-SHAP at commit ebcaa32853b4465a16205a0f5f15ce76f71c1ea3. Retains the original notebook and dataset. New workflows live in `credit_risk/`; the API now reads newly trained bundles instead of the legacy pickle files.

## Start in VS Code (Windows, Python 3.12)
Open this project folder and run in the terminal:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m credit_risk.eda
.venv\Scripts\python -m credit_risk.train
.venv\Scripts\python -m uvicorn main:app --reload
```
Open http://127.0.0.1:8000 for the existing interface with a model selector. API documentation: http://127.0.0.1:8000/docs.
On macOS/Linux use `python3.12 -m venv .venv` and `.venv/bin/python` instead.
If the delivered folder already contains `artifacts/manifest.json`, the supplied trained run is ready to serve; skip training or choose a new `--output` directory.

## Update: two-dataset placement benchmark

Both the original loan dataset and UCI Default of Credit Card Clients are supported. Read `reports/dataset_comparison/COMPARISON.md` and `all_metrics.csv` for the measured comparison. At the standard 0.5 threshold, original XGBoost reached 93.58% test accuracy; UCI models reached 81.45–82.36% in this baseline run. These are different tasks, so that gap is not a model improvement claim.

The supplied archive contains trained bundles for both datasets. After installing dependencies, run the server as above and visit:
- http://127.0.0.1:8000 — original loan form.
- http://127.0.0.1:8000/uci — credit-card form with five model choices and two operating points.
- http://127.0.0.1:8000/docs — both API schemas.

Reproduce the UCI workflow (use a new output directory if artifacts already exist):

```powershell
.venv\Scripts\python -m credit_risk.fetch_uci
.venv\Scripts\python -m credit_risk.uci_eda
.venv\Scripts\python -m credit_risk.train --dataset uci --output artifacts_uci_new
.venv\Scripts\python -m credit_risk.explain --artifacts artifacts_uci --rows 5
.venv\Scripts\python -m credit_risk.compare_datasets
```

The last two commands inspect the supplied `artifacts_uci` run. To serve your new run set `$env:UCI_ARTIFACTS="artifacts_uci_new"` before starting the server. The comparison command expects the canonical `artifacts` and `artifacts_uci` directories.

The UCI adapter excludes customer ID, treats demographic/repayment codes as categorical and preserves undocumented codes without inventing meanings. Exact duplicate feature+label rows are removed. Identical feature profiles are grouped into the same split, including profiles with conflicting labels. Optional tuning/calibration folds also respect these groups. The split is approximately 60/20/20 rather than exact, because groups stay intact.

The source CSV and ZIP are included, with attribution in `data/uci/README.md`. Historical Taiwan amounts are NT dollars. The UCI target is next-month credit-card default, not a new-loan approval outcome. `loan_status` is merely a shared internal target column name.

Both default APIs still use validation-average-precision model selection and validation-F1 thresholds. The report separately provides standard-0.5 accuracy and models chosen using validation accuracy. Do not mix metrics from different operating points in resume claims. Eight automated tests check both APIs and split isolation. The full UCI tuning/calibration option and hosting have not been executed.

## Modules

| File | Responsibility |
|---|---|
| `credit_risk/data.py` | Schema, fixed cleaning rules, shared train/validation/test split |
| `credit_risk/models.py` | Preprocessing, five model definitions, tuning spaces |
| `credit_risk/train.py` | Training, optional tuning/calibration, selection and saving |
| `credit_risk/evaluation.py` | Correct F1 threshold optimization and metrics |
| `credit_risk/eda.py` | Quality audit, distributions, missingness, subgroup default rates |
| `credit_risk/explain.py` | SHAP for the exact saved prediction pipeline |
| `main.py` | Validated FastAPI inference and model listing |

Models: Logistic Regression, Random Forest, AdaBoost, Gradient Boosting, XGBoost. Defaults are unweighted so the baseline experiment is easy to reproduce. Imbalance is assessed using average precision, recall and precision, rather than accuracy alone. Class weighting/resampling should be evaluated as a separate experiment inside training folds.

## Experiment design

1. Remove exact duplicate records and apply documented fixed validity rules. Missing predictors remain for imputation. These rules define an adult-applicant cohort and should be revisited against the data dictionary.
2. Split into 60% training, 20% validation, 20% test using the same indices for every model.
3. Fit preprocessing only on training data (inside each fold during optional tuning).
4. Select the model using validation average precision. Optimize each model's F1 threshold using validation predictions only.
5. Freeze those choices, then report test metrics for all models. Do not use the test comparison to repeatedly select new models.
6. Save each exact fitted pipeline and its matching threshold together. Manifest records versions, seed and split indices. No refit occurs after threshold selection.

Average precision is the reported precision-recall summary, not trapezoidal PR-AUC. Brier score assesses probability error. Default uncalibrated probabilities are not established estimates of real-world lending risk.

```powershell
# More thorough separate experiment; may take several minutes
.venv\Scripts\python -m credit_risk.train --tune --calibrate --output artifacts_tuned
# Explain saved default model or choose another
.venv\Scripts\python -m credit_risk.explain --rows 20
.venv\Scripts\python -m credit_risk.explain --model adaboost --rows 5
# Serve the alternate experiment
$env:CREDIT_RISK_ARTIFACTS="artifacts_tuned"
.venv\Scripts\python -m uvicorn main:app --reload
```

SHAP uses model-agnostic permutation explanations of the saved class-1 probability, so it also works with AdaBoost and calibrated estimators. It is slower than tree-specific explainers. Background rows come from training data; examples come from test data. The bar plot summarizes the explained sample only. Keep the input CSV unchanged after training because stored indices identify rows. SHAP is associative, not a causal explanation.

## Expanded EDA

`reports/eda/` contains raw quality counts, missingness, class balance, numeric histograms and boxplots by target, correlations, category sample counts and default rates, quantile-bin risk profiles, and loan-grade × loan-intent heatmaps with supporting counts. Statistical exploration uses the training split to preserve the final holdout.

Possible original extensions, to choose based on your research question:
- Compare with/without `loan_grade` and `loan_int_rate` after checking prediction-time availability.
- Add an income-to-loan feature ablation; `loan_percent_income` already represents the inverse relationship, so avoid presenting the same signal as a new independent feature.
- Compare sigmoid calibration and uncalibrated probabilities using Brier scores.
- Choose a threshold using explicit false-negative/false-positive costs, rather than F1, once justified costs exist.
- Analyze errors and subgroup recall with sample counts and uncertainty; use a fresh holdout for subsequent model revisions.

No verified borrower identifier or observation date is available in the supplied columns. A random split does not establish temporal generalization. Confirm source, license, units and the target definition before publishing findings. The original rupee label was removed because dataset currency has not been verified.

## Issues corrected from the original notebook

- F1 threshold formula used subtraction instead of precision × recall.
- Threshold was optimized on test labels.
- Saved calibrated model was paired with a threshold from another model.
- Missing employment values were unintentionally removed by comparisons.
- XGBoost's CV configuration used `n_estimator` instead of `n_estimators`.

The original notebook is retained for reference; execute the new modules for corrected results. Legacy `credit_risk_model.pkl` and `best_threshold.pkl` are not loaded by the new API.

## Verification

```powershell
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest -q
```
The API integration test requires all five default bundles in `artifacts/`. Tests cover threshold correctness, missing-value handling, disjoint splits, unknown categories, each model's endpoint and input validation. Only load trusted joblib files.

The Render configuration trains into `deployment_artifacts` during the build and serves that run. Build resource limits may require a smaller model set or prebuilt artifacts. Hosting has not been tested or deployed by this update.
