"""Native JSON for uncalibrated XGBoost; joblib for preprocessing/other estimators."""
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline


def export_bundle(directory, name):
    directory = Path(directory)
    bundle = joblib.load(directory / f'{name}.joblib')
    model = bundle['model']
    from xgboost import XGBClassifier
    if isinstance(model, Pipeline) and isinstance(model.named_steps.get('classifier'), XGBClassifier):
        destination = directory / f'{name}.model.json'
        model.named_steps['classifier'].save_model(destination)
        portable = {k: v for k, v in bundle.items() if k != 'model'}
        portable['preprocessor'] = model.named_steps['preprocessor']
        portable['xgboost_json'] = destination.name
        joblib.dump(portable, directory / f'{name}.portable.joblib')
        return destination
    return None


def load_bundle(directory, name):
    directory = Path(directory)
    portable_path = directory / f'{name}.portable.joblib'
    if not portable_path.exists():
        return joblib.load(directory / f'{name}.joblib')
    bundle = joblib.load(portable_path)
    from xgboost import XGBClassifier
    classifier = XGBClassifier()
    classifier.load_model(directory / bundle.pop('xgboost_json'))
    bundle['model'] = Pipeline([('preprocessor', bundle.pop('preprocessor')), ('classifier', classifier)])
    return bundle
