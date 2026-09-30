from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = 'loan_status'
NUMERIC = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
           'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
CATEGORICAL = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
FEATURES = NUMERIC + CATEGORICAL


def load_data(path):
    df = pd.read_csv(Path(path))
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    df = df[FEATURES + [TARGET]].copy()
    if df[TARGET].isna().any() or set(df[TARGET].unique()) != {0, 1}:
        raise ValueError('loan_status must contain both 0 and 1, with no missing targets')
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors='raise')
    return df


def clean_data(df):
    """Fixed domain rules; preserve missing predictors for training-fold imputation."""
    df = df.drop_duplicates().copy()
    valid = pd.Series(True, index=df.index)
    rules = {
        'person_age': df.person_age.between(18, 100),
        'person_income': df.person_income.gt(0),
        'loan_amnt': df.loan_amnt.gt(0),
        'person_emp_length': df.person_emp_length.between(0, 60),
        'loan_int_rate': df.loan_int_rate.ge(0),
        'loan_percent_income': df.loan_percent_income.ge(0),
        'cb_person_cred_hist_length': df.cb_person_cred_hist_length.ge(0),
    }
    for col, rule in rules.items():
        valid &= df[col].isna() | rule
    valid &= df.person_emp_length.isna() | df.person_age.isna() | df.person_emp_length.le(df.person_age)
    return df.loc[valid].copy()


def split_data(df, seed=42):
    """60% train, 20% validation, 20% test; shared by every model."""
    train_val, test = train_test_split(df, test_size=.2, stratify=df[TARGET], random_state=seed)
    train, val = train_test_split(train_val, test_size=.25, stratify=train_val[TARGET], random_state=seed)
    return train, val, test
