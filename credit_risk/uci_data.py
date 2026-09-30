"""UCI 350 adapter. Preserve category codes; never use customer ID as a predictor."""
from pathlib import Path
import pandas as pd

TARGET = 'loan_status'  # Internal API name; source meaning is next-month card default.
PAY_STATUS = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
CATEGORICAL = ['SEX', 'EDUCATION', 'MARRIAGE'] + PAY_STATUS
NUMERIC = ['LIMIT_BAL', 'AGE'] + [f'BILL_AMT{i}' for i in range(1, 7)] + [f'PAY_AMT{i}' for i in range(1, 7)]
FEATURES = NUMERIC + CATEGORICAL


def load_data(path):
    raw = pd.read_csv(Path(path))
    df = raw.rename(columns={'default payment next month': TARGET})
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f'Missing UCI columns: {sorted(missing)}')
    df = df[FEATURES + [TARGET]].copy()
    if df[TARGET].isna().any() or set(df[TARGET].unique()) != {0, 1}:
        raise ValueError('Expected binary default target')
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors='raise')
    for col in CATEGORICAL:
        # Discrete codes are categories, including undocumented 0/-2/etc codes.
        df[col] = df[col].map(lambda x: str(int(x)) if pd.notna(x) else float('nan'))
    return df


def clean_data(df):
    # No loan-application age/income rules are transferred to this different task.
    return df.drop_duplicates().copy()


def split_data(df, seed=42):
    """Approximately 60/20/20, keeping identical predictor profiles together."""
    from sklearn.model_selection import StratifiedGroupKFold
    groups = pd.util.hash_pandas_object(df[FEATURES], index=False)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    dev_idx, test_idx = next(outer.split(df, df[TARGET], groups))
    dev, test = df.iloc[dev_idx], df.iloc[test_idx]
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    train_idx, val_idx = next(inner.split(dev, dev[TARGET], groups.iloc[dev_idx]))
    return dev.iloc[train_idx], dev.iloc[val_idx], test
