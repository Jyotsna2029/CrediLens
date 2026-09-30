# CrediLens live SHAP update

## Apply to your existing Windows project

1. Stop Uvicorn with Ctrl+C.
2. Extract this update ZIP into the existing project folder containing main.py. Allow replacement of the listed code files. Merge the credit_risk and static folders; do not delete those folders.
3. The update contains no datasets, model artifacts or requirements file. Keep artifacts_windows and artifacts_uci_windows, your existing CSVs, and .venv.
4. From the project terminal run:

```powershell
.\.venv\Scripts\python.exe run_credilens.py
```

The launcher chooses artifacts_windows and artifacts_uci_windows when available, unless you explicitly set CREDIT_RISK_ARTIFACTS or UCI_ARTIFACTS. It prints the chosen paths, exports locally loadable XGBoost pipelines once to native model JSON, and starts Uvicorn. If export reports an error, stop and check that those paths point to the locally trained directories.

5. Wait for Application startup complete, open http://127.0.0.1:8000, and press Ctrl+F5.
6. Submit an application, then click Explain this prediction. The first explanation may be slower while SHAP initializes. Change the inputs or model and assess again to get a new explanation.

This command runs directly in the terminal. Do not paste PowerShell script-file code using $PSScriptRoot.

## What changed

- Added POST /explain for the exact original-dataset model selected in the UI.
- 11 input-feature SHAP contributions in probability space, displayed in percentage points.
- Positive/red raises the model estimate; negative/green lowers it.
- Background uses 32 fixed training rows, verified against the training dataset hash. No test rows are used as the explanation background.
- The backend verifies base value + all contributions = the predicted probability.
- Explanations are model-agnostic permutation estimates (five permutations), not exact causal explanations. Related features can share effects; independent masking can create unusual feature combinations.
- Explains submitted input snapshots; editing the form hides stale predictions/explanations.
- Clearer validation messages, browser form validation, range warnings within the explanation, CrediLens branding, rupee display labels and hidden automatic ratio retained.
- UCI prediction interface remains available; live UCI SHAP is not added in this update.
- The currency labels do not convert inputs or establish suitability for Indian lending. Dataset currency/provenance must be resolved before making that claim.

## Model export and deployment

The new loader prefers <model>.portable.joblib plus <model>.model.json for uncalibrated XGBoost pipelines. Existing .joblib files are retained. Native JSON avoids pickling the XGBoost booster for transfer. Other estimators and fitted preprocessing still use joblib and need compatible dependency versions.
The export must be run on the machine where the original model loads (your Windows machine). Exporting these same already-invalid snapshots on a different machine cannot repair them. Calibrated XGBoost ensembles do not receive native export support in this update.
For hosting, copy the manifest, all model files including native JSON and portable sidecars, and the unchanged original CSV needed for SHAP. Use an appropriate production startup command without --reload. This update has not been deployed publicly or tested on a hosting provider.

## Verification

11 automated tests passed: all five prediction/explanation pairs, SHAP additivity, input validation, missing predictors, out-of-range detection, grouped UCI splits, original/UCI coexistence, and native XGBoost prediction equivalence. JavaScript syntax checked.

Actual Uvicorn startup, health endpoint and updated HTML serving verified. Automated browser rendering was unavailable because the Chromium download failed; visual layout must be checked on the user machine.
