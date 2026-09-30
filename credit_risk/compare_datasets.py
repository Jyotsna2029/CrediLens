"""Reproduce a descriptive comparison from frozen model predictions."""
from pathlib import Path
import json
import math
import joblib
import pandas as pd
from .evaluation import evaluate
from .data import load_data, FEATURES, TARGET


def wilson(correct,n):
    p=correct/n;z=1.96
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return center-half,center+half


def main():
    out=Path('reports/dataset_comparison');out.mkdir(parents=True,exist_ok=True)
    rows=[];selection=[]
    for dataset,directory in [('Existing loan dataset','artifacts'),('UCI credit card dataset','artifacts_uci')]:
        root=Path(directory);manifest=json.loads((root/'manifest.json').read_text())
        original=directory=='artifacts'
        if original:
            df=load_data('credit_risk_dataset.csv')
            val=df.loc[manifest['source_rows']['validation']]
        for name in manifest['models']:
            predictions=pd.read_csv(root/f'{name}_test_predictions.csv')
            bundle=joblib.load(root/f'{name}.joblib')
            if original:
                vprob=bundle['model'].predict_proba(val[FEATURES])[:,1]
                val_accuracy=evaluate(val[TARGET],vprob,.5)['accuracy']
            else:
                comp=pd.read_csv(root/'model_comparison.csv').set_index('model')
                val_accuracy=float(comp.loc[name,'validation_accuracy_at_05'])
            selection.append(dict(dataset=dataset,model=name,validation_accuracy_at_05=val_accuracy))
            y=predictions.loan_status;p=predictions.probability
            baseline=max(y.mean(),1-y.mean())
            for operating_point,threshold in [('Standard 0.5',.5),('Validation F1',bundle['threshold'])]:
                metrics=evaluate(y,p,threshold)
                low,high=wilson(metrics['tp']+metrics['tn'],len(y))
                rows.append(dict(dataset=dataset,model=name,operating_point=operating_point,threshold=threshold,
                    test_rows=len(y),default_rate=float(y.mean()),majority_baseline=float(baseline),
                    accuracy_ci_lower=low,accuracy_ci_upper=high,**metrics))
    results=pd.DataFrame(rows);results.to_csv(out/'all_metrics.csv',index=False)
    pd.DataFrame(selection).to_csv(out/'validation_accuracy_selection.csv',index=False)
    lines=['# Dataset comparison: frozen baseline models','','This is a descriptive comparison of two different prediction tasks, not evidence that one dataset/model generalizes better. Model settings were unchanged; no tuning/calibration used. Original split is stratified random; UCI split is stratified by identical-feature groups to prevent profile overlap. Both are approximately 60/20/20.','',
    'An initial UCI random-split run was invalidated after discovering identical predictor profiles across partitions. Only the corrected grouped run is reported. No hyperparameters or thresholds were selected from test scores. The original test results were already observed before this dataset comparison, so this is not a new blind external validation.','',
    '## Accuracy at the standard 0.5 threshold','', '| Model | Existing loan data | UCI credit cards |','|---|---:|---:|']
    for model in results.model.unique():
        r=results[(results.model==model)&(results.operating_point=='Standard 0.5')].set_index('dataset')
        lines.append(f"| {model} | {r.loc['Existing loan dataset','accuracy']:.2%} | {r.loc['UCI credit card dataset','accuracy']:.2%} |")
    lines+=['','## Validation-selected models','']
    for dataset in results.dataset.unique():
        candidates=[r for r in selection if r['dataset']==dataset]
        chosen=max(candidates,key=lambda r:r['validation_accuracy_at_05'])
        metric=results[(results.dataset==dataset)&(results.model==chosen['model'])&(results.operating_point=='Standard 0.5')].iloc[0]
        lines.append(f"- {dataset}: select {chosen['model']} using validation accuracy {chosen['validation_accuracy_at_05']:.2%}; test accuracy {metric.accuracy:.2%}, F1 {metric.f1:.3f}, recall {metric.recall:.2%}, ROC-AUC {metric.roc_auc:.3f}. Majority baseline {metric.majority_baseline:.2%}.")
    lines+=['','The serving defaults remain selected by validation average precision. The API F1 operating point uses each exact model’s validation-selected F1 threshold. Standard-threshold results must not be described as F1-threshold results.','',
    '## Resume interpretation','','Name the dataset and holdout protocol when reporting accuracy, and add ROC-AUC or default recall. Do not call these cross-dataset accuracy differences an improvement of one model over another. A different target and cohort changes task difficulty. These numbers are baseline results, not optimized limits on achievable performance.','',
    'UCI source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients (Yeh, 2009; DOI 10.24432/C55S3H; CC BY 4.0).',
    'The existing CSV originates from the supplied GitHub repository; its upstream provenance and license remain unverified. The UCI benchmark has clearer source documentation.',
    'Wilson intervals in all_metrics.csv treat test records as independent; identical-profile clusters in UCI mean these are approximate, not cluster-adjusted intervals.']
    (out/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__': main()
