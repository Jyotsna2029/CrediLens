"""Local, probability-space SHAP explanations of the exact served pipeline."""
import hashlib
import threading
import numpy as np
import pandas as pd
from .data import FEATURES, CATEGORICAL, NUMERIC, load_data

LABELS = dict(person_age='Age', person_income='Annual income', person_home_ownership='Home ownership',
    person_emp_length='Employment length', loan_intent='Loan purpose', loan_grade='Loan grade',
    loan_amnt='Loan amount', loan_int_rate='Interest rate', loan_percent_income='Loan-to-income ratio',
    cb_person_default_on_file='Prior default', cb_person_cred_hist_length='Credit history length')


class ExplanationService:
    def __init__(self, data_path, manifest):
        if hashlib.sha256(data_path.read_bytes()).hexdigest() != manifest['dataset_sha256']:
            raise ValueError('Explanation data does not match this training run')
        df = load_data(data_path)
        training = df.loc[manifest['source_rows']['train'], FEATURES]
        self.background = training.sample(min(32, len(training)), random_state=42).copy()
        self.categories = {c: list(training[c].dropna().unique()) for c in CATEGORICAL}
        self.bounds = {c: (float(training[c].min()), float(training[c].max())) for c in NUMERIC}
        self.lock = threading.Lock()

    def explain(self, model, frame):
        import shap
        # A non-blocking lock limits expensive requests and controls SHAP's RNG.
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError('Another explanation is running. Please try again shortly.')
        try:
            categories = {c: list(v) for c, v in self.categories.items()}
            for c in CATEGORICAL:
                value = frame.iloc[0][c]
                if pd.notna(value) and value not in categories[c]:
                    categories[c].append(value)
            def encode(raw):
                data = raw[FEATURES].copy()
                for c, vocabulary in categories.items():
                    data[c] = pd.Categorical(data[c], categories=vocabulary).codes.astype(float)
                return data.astype(float)
            def predict(encoded):
                data = pd.DataFrame(encoded, columns=FEATURES)
                for c, vocabulary in categories.items():
                    data[c] = [vocabulary[int(v)] if 0 <= int(v) < len(vocabulary) else np.nan for v in data[c]]
                return model.predict_proba(data)[:, 1]
            background = encode(self.background)
            x = encode(frame)
            explainer = shap.Explainer(predict, shap.maskers.Independent(background),
                algorithm='permutation', feature_names=FEATURES, seed=42)
            explanation = explainer(x, max_evals=5*(2*len(FEATURES)+1))
            contributions = explanation.values[0]
            base = float(explanation.base_values[0])
            probability = float(model.predict_proba(frame)[:, 1][0])
            residual = probability - (base + float(contributions.sum()))
            if not np.isfinite(contributions).all() or abs(residual) > 1e-5:
                raise ValueError('Explanation consistency check failed')
            features = []
            for col, contribution in zip(FEATURES, contributions):
                value = frame.iloc[0][col]
                value = None if pd.isna(value) else (value.item() if isinstance(value, np.generic) else value)
                features.append(dict(feature=col, label=LABELS[col], value=value,
                    contribution=float(contribution), percentage_points=float(100*contribution),
                    direction='raises' if contribution > 0 else ('lowers' if contribution < 0 else 'neutral')))
            features.sort(key=lambda f: abs(f['contribution']), reverse=True)
            warnings = []
            for col, (lo, hi) in self.bounds.items():
                value = frame.iloc[0][col]
                if pd.notna(value) and not lo <= value <= hi:
                    warnings.append(f'{LABELS[col]} is outside the training range ({lo:g} to {hi:g}); this prediction may be unreliable.')
            return dict(method='Permutation SHAP', output_space='default_probability',
                base_probability=base, default_probability=probability, features=features,
                additivity_residual=residual, background_rows=len(self.background), warnings=warnings,
                note='Contributions describe the model, not causes. Correlated features, including loan amount, income and their ratio, can share effects. The background average is not the population default rate.')
        finally:
            self.lock.release()
