# Dataset comparison: frozen baseline models

This is a descriptive comparison of two different prediction tasks, not evidence that one dataset/model generalizes better. Model settings were unchanged; no tuning/calibration used. Original split is stratified random; UCI split is stratified by identical-feature groups to prevent profile overlap. Both are approximately 60/20/20.

An initial UCI random-split run was invalidated after discovering identical predictor profiles across partitions. Only the corrected grouped run is reported. No hyperparameters or thresholds were selected from test scores. The original test results were already observed before this dataset comparison, so this is not a new blind external validation.

## Accuracy at the standard 0.5 threshold

| Model | Existing loan data | UCI credit cards |
|---|---:|---:|
| logistic_regression | 86.50% | 82.03% |
| random_forest | 93.54% | 82.36% |
| adaboost | 88.04% | 81.45% |
| gradient_boosting | 92.52% | 82.13% |
| xgboost | 93.58% | 82.21% |

## Validation-selected models

- Existing loan dataset: select xgboost using validation accuracy 92.93%; test accuracy 93.58%, F1 0.833, recall 72.92%, ROC-AUC 0.942. Majority baseline 78.12%.
- UCI credit card dataset: select logistic_regression using validation accuracy 81.98%; test accuracy 82.03%, F1 0.473, recall 36.43%, ROC-AUC 0.771. Majority baseline 77.87%.

The serving defaults remain selected by validation average precision. The API F1 operating point uses each exact model’s validation-selected F1 threshold. Standard-threshold results must not be described as F1-threshold results.

## Resume interpretation

Name the dataset and holdout protocol when reporting accuracy, and add ROC-AUC or default recall. Do not call these cross-dataset accuracy differences an improvement of one model over another. A different target and cohort changes task difficulty. These numbers are baseline results, not optimized limits on achievable performance.

UCI source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients (Yeh, 2009; DOI 10.24432/C55S3H; CC BY 4.0).
The existing CSV originates from the supplied GitHub repository; its upstream provenance and license remain unverified. The UCI benchmark has clearer source documentation.
Wilson intervals in all_metrics.csv treat test records as independent; identical-profile clusters in UCI mean these are approximate, not cluster-adjusted intervals.
