# Baseline results

Seed 42; shared 60/20/20 split. Model selected on validation average precision; thresholds selected on validation F1. No tuning or calibration enabled.

| model | test_roc_auc | test_average_precision | test_f1 | test_recall |
| --- | --- | --- | --- | --- |
| xgboost | 0.9419 | 0.8963 | 0.8334 | 0.7426 |
| random_forest | 0.9305 | 0.8853 | 0.8268 | 0.7306 |
| gradient_boosting | 0.9267 | 0.8733 | 0.8055 | 0.7172 |
| adaboost | 0.8923 | 0.7680 | 0.7024 | 0.7355 |
| logistic_regression | 0.8699 | 0.7178 | 0.6758 | 0.6608 |

Five automated tests passed. Permutation SHAP and its additivity check passed for AdaBoost and XGBoost on two held-out examples each. Optional full tuning/calibration and hosting have not been run.

The original notebook is legacy reference material and retains the issues documented in README.md. Use the new modules for corrected experiments.

## UCI comparison added
See reports/dataset_comparison/COMPARISON.md for standard-threshold accuracies across both datasets and all_metrics.csv for both operating points. The earlier table above describes the original validation-F1 operating point.
