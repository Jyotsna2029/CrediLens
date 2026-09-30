"""Run from repository root: python -m credit_risk.train"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import importlib.metadata
import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score
from .data import split_data
from . import data as original_data
from . import uci_data
from .models import make_model, MODEL_NAMES, PARAMETERS
from .evaluation import choose_threshold, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['original', 'uci'], default='original')
    parser.add_argument('--data', default=None)
    parser.add_argument('--output', default='artifacts')
    parser.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tune', action='store_true', help='Training-only 3-fold randomized search')
    parser.add_argument('--calibrate', action='store_true', help='Training-only 3-fold sigmoid calibration')
    args = parser.parse_args()
    adapter = original_data if args.dataset == 'original' else uci_data
    FEATURES, TARGET = adapter.FEATURES, adapter.TARGET
    args.data = args.data or ('credit_risk_dataset.csv' if args.dataset == 'original' else 'data/uci/credit_card_clients.csv')
    out = Path(args.output)
    if (out / 'manifest.json').exists():
        parser.error('Output already has a trained run. Choose a new --output directory.')
    out.mkdir(parents=True, exist_ok=True)
    raw = adapter.load_data(args.data)
    cleaned = adapter.clean_data(raw)
    train, val, test = (uci_data.split_data(cleaned, args.seed) if args.dataset == 'uci' else split_data(cleaned, args.seed))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    if args.dataset == 'uci':
        from sklearn.model_selection import StratifiedGroupKFold
        groups = pd.util.hash_pandas_object(train[FEATURES], index=False)
        cv = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=args.seed).split(train, train[TARGET], groups))
    trained, rows = {}, []
    for name in args.models:
        print(f'Training {name}...', flush=True)
        model = make_model(name, args.seed, adapter.NUMERIC, adapter.CATEGORICAL)
        if args.tune:
            search = RandomizedSearchCV(model, PARAMETERS[name], n_iter=4, cv=cv,
                scoring='average_precision', random_state=args.seed, n_jobs=2, error_score='raise')
            search.fit(train[FEATURES], train[TARGET])
            model = search.best_estimator_
        if args.calibrate:
            model = CalibratedClassifierCV(model, method='sigmoid', cv=cv)
        model.fit(train[FEATURES], train[TARGET])
        p = model.predict_proba(val[FEATURES])[:, 1]
        threshold = choose_threshold(val[TARGET], p)
        trained[name] = {'model': model, 'threshold': threshold, 'features': FEATURES, 'dataset': args.dataset}
        rows.append({'model': name, 'threshold': threshold, 'validation_average_precision': float(average_precision_score(val[TARGET], p)), 'validation_accuracy_at_05': evaluate(val[TARGET], p, .5)['accuracy']})
    # Freeze winner before looking at test labels or scores.
    winner = max(rows, key=lambda row: row['validation_average_precision'])['model']
    for row in rows:
        name = row['model']
        bundle = trained[name]
        joblib.dump(bundle, out / f'{name}.joblib')
        p = bundle['model'].predict_proba(test[FEATURES])[:, 1]
        row.update({f'test_{key}': value for key, value in evaluate(test[TARGET], p, bundle['threshold']).items()})
        row.update({f'test_at_05_{key}': value for key, value in evaluate(test[TARGET], p, .5).items()})
        predictions = test.copy()
        predictions['probability'] = p
        predictions['prediction'] = (p >= bundle['threshold']).astype(int)
        predictions['error_type'] = ['correct' if a == b else ('false_negative' if a == 1 else 'false_positive') for a, b in zip(test[TARGET], predictions.prediction)]
        predictions.to_csv(out / f'{name}_test_predictions.csv', index_label='source_row')
    comparison = pd.DataFrame(rows).sort_values('validation_average_precision', ascending=False)
    comparison.to_csv(out / 'model_comparison.csv', index=False)
    versions = {}
    for package in ['pandas', 'scikit-learn', 'xgboost', 'joblib']:
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    manifest = dict(dataset=args.dataset, split_method=('stratified_group_by_identical_features' if args.dataset == 'uci' else 'stratified_random'), numeric_features=adapter.NUMERIC, categorical_features=adapter.CATEGORICAL, dataset_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(), default_model=winner, models=list(trained), seed=args.seed,
        calibrated=args.calibrate, tuned=args.tune, python=platform.python_version(), versions=versions,
        raw_rows=len(raw), cleaned_rows=len(cleaned), train_rows=len(train), validation_rows=len(val), test_rows=len(test),
        selection_metric='validation_average_precision', threshold_objective='validation_F1',
        positive_class=('loan_status=1 (default)' if args.dataset == 'original' else 'default payment next month=1'),
        test_default_rate=float(test[TARGET].mean()), majority_class_baseline_accuracy=float(max(test[TARGET].mean(), 1-test[TARGET].mean())),
        accuracy_at_05_selected_model=max(rows, key=lambda row: row['validation_accuracy_at_05'])['model'],
        source_rows={name: frame.index.tolist() for name, frame in [('train', train), ('validation', val), ('test', test)]})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(comparison.to_string(index=False))
    print(f'Default model selected on validation data: {winner}')


if __name__ == '__main__':
    main()
