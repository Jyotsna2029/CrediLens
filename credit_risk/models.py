from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from .data import NUMERIC, CATEGORICAL

MODEL_NAMES = ['logistic_regression', 'random_forest', 'adaboost', 'gradient_boosting', 'xgboost']


def make_model(name, seed=42, numeric_cols=None, categorical_cols=None):
    numeric_cols = NUMERIC if numeric_cols is None else numeric_cols
    categorical_cols = CATEGORICAL if categorical_cols is None else categorical_cols
    numeric = [('imputer', SimpleImputer(strategy='median', keep_empty_features=True))]
    if name == 'logistic_regression':
        numeric.append(('scaler', StandardScaler()))
    preprocessing = ColumnTransformer([
        ('numeric', Pipeline(numeric), numeric_cols),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('encoding', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_cols)
    ])
    models = {
        'logistic_regression': lambda: LogisticRegression(max_iter=2000, random_state=seed),
        'random_forest': lambda: RandomForestClassifier(n_estimators=250, min_samples_leaf=2, n_jobs=2, random_state=seed),
        'adaboost': lambda: AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=seed), n_estimators=150, learning_rate=.5, random_state=seed),
        'gradient_boosting': lambda: GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=.05, random_state=seed),
    }
    if name == 'xgboost':
        from xgboost import XGBClassifier
        classifier = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=.05,
                                   subsample=.8, colsample_bytree=.8, eval_metric='logloss',
                                   tree_method='hist', n_jobs=2, random_state=seed)
    elif name in models:
        classifier = models[name]()
    else:
        raise ValueError(f'Unknown model: {name}')
    return Pipeline([('preprocessor', preprocessing), ('classifier', classifier)])


PARAMETERS = {
    'logistic_regression': {'classifier__C': [.01, .1, 1, 10]},
    'random_forest': {'classifier__max_depth': [None, 8, 16], 'classifier__min_samples_leaf': [1, 3, 6]},
    'adaboost': {'classifier__n_estimators': [100, 200], 'classifier__learning_rate': [.1, .5, 1]},
    'gradient_boosting': {'classifier__max_depth': [2, 3, 4], 'classifier__learning_rate': [.03, .1]},
    'xgboost': {'classifier__max_depth': [3, 5, 7], 'classifier__learning_rate': [.03, .1], 'classifier__n_estimators': [150, 300]},
}
