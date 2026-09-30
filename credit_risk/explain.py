"""Explain the saved model's class-1 probability, including calibrated models."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from . import data as original_data
from . import uci_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', default='artifacts')
    parser.add_argument('--data', default=None)
    parser.add_argument('--model', default=None)
    parser.add_argument('--rows', type=int, default=5)
    args = parser.parse_args()
    if args.rows < 1:
        parser.error('--rows must be positive')
    root = Path(args.artifacts)
    manifest = json.loads((root/'manifest.json').read_text())
    adapter = uci_data if manifest.get('dataset') == 'uci' else original_data
    FEATURES, NUMERIC = adapter.FEATURES, adapter.NUMERIC
    args.data = args.data or ('data/uci/credit_card_clients.csv' if manifest.get('dataset') == 'uci' else 'credit_risk_dataset.csv')
    name = args.model or manifest['default_model']
    if name not in manifest['models']:
        parser.error('Requested model is not in this run')
    bundle = joblib.load(root/f'{name}.joblib')
    if hashlib.sha256(Path(args.data).read_bytes()).hexdigest() != manifest['dataset_sha256']:
        raise ValueError('Dataset differs from the training run')
    df = adapter.load_data(args.data)
    train = df.loc[manifest['source_rows']['train'], FEATURES]
    test = df.loc[manifest['source_rows']['test'], FEATURES].head(args.rows)
    # Encode categories only for the masker. Decode inside prediction so SHAP
    # explains the exact saved pipeline, including preprocessing/calibration.
    categories = {c: list(df[c].dropna().unique()) for c in FEATURES if c not in NUMERIC}
    def encode(frame):
        frame = frame.copy()
        for col, values in categories.items():
            frame[col] = pd.Categorical(frame[col], categories=values).codes.astype(float)
        return frame.astype(float)
    def predict(encoded):
        frame = pd.DataFrame(encoded, columns=FEATURES)
        for col, values in categories.items():
            frame[col] = [values[int(v)] if 0 <= int(v) < len(values) else np.nan for v in frame[col]]
        return bundle['model'].predict_proba(frame)[:, 1]
    background = encode(train.sample(min(40,len(train)),random_state=manifest['seed']))
    examples = encode(test)
    explainer = shap.Explainer(predict, shap.maskers.Independent(background), algorithm='permutation', feature_names=FEATURES, seed=manifest['seed'])
    values = explainer(examples, max_evals=10*(2*len(FEATURES)+1))
    actual = predict(examples.to_numpy())
    if not np.allclose(values.base_values + values.values.sum(axis=1), actual, atol=1e-5):
        raise RuntimeError('SHAP additivity check failed')
    out = root/'explanations'/name
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(values.values,columns=FEATURES,index=test.index).to_csv(out/'shap_values.csv')
    pd.DataFrame({'base_probability':values.base_values,'predicted_probability':actual},index=test.index).to_csv(out/'probabilities.csv')
    shap.plots.bar(values,show=False)
    plt.tight_layout(); plt.savefig(out/'importance.png',dpi=140,bbox_inches='tight'); plt.close()
    shap.plots.waterfall(values[0],show=False)
    plt.savefig(out/'applicant_waterfall.png',dpi=140,bbox_inches='tight'); plt.close()
    (out/'README.txt').write_text('Permutation SHAP explains the saved model probability. Importance summarizes only the requested rows, not the full population. Categorical waterfall values are encoded category indices; consult input data for original values. Contributions are associative, not causal. Correlated features can share credit. Use the same unchanged dataset used for training.\n')
    print(f'Explanations saved to {out}')

if __name__ == '__main__':
    main()
