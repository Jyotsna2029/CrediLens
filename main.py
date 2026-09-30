"""Serve locally trained bundles; run python -m credit_risk.train first."""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
import json
import os
import logging
from credit_risk.persistence import load_bundle
from credit_risk.live_explanations import ExplanationService
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from credit_risk.uci_api import router as uci_router, load_models as load_uci_models, loaded as uci_loaded

ROOT = Path(__file__).resolve().parent
ml_model = {}

@asynccontextmanager
async def lifespan(app):
    directory = Path(os.environ.get('CREDIT_RISK_ARTIFACTS', str(ROOT / 'artifacts')))
    manifest_path = directory / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        ml_model['manifest'] = manifest
        ml_model['models'] = {name: load_bundle(directory, name) for name in manifest['models']}
        try:
            ml_model['explanation_service'] = ExplanationService(ROOT / 'credit_risk_dataset.csv', manifest)
        except (OSError, ValueError, KeyError):
            logging.exception('Live explanations unavailable: check dataset and manifest')
    load_uci_models(ROOT)
    yield
    uci_loaded.clear()
    ml_model.clear()

app = FastAPI(title='CrediLens — Explainable Credit Risk', lifespan=lifespan)

class LoanApplication(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra='forbid')
    person_age: int = Field(ge=18, le=100)
    person_income: float = Field(gt=0)
    person_home_ownership: Literal['RENT', 'MORTGAGE', 'OWN', 'OTHER']
    person_emp_length: float | None = Field(default=None, ge=0, le=60)
    loan_intent: Literal['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']
    loan_grade: Literal['A', 'B', 'C', 'D', 'E', 'F', 'G']
    loan_amnt: float = Field(gt=0)
    loan_int_rate: float | None = Field(default=None, ge=0)
    loan_percent_income: float = Field(ge=0)
    cb_person_default_on_file: Literal['Y', 'N']
    cb_person_cred_hist_length: int = Field(ge=0)

    @model_validator(mode='after')
    def validate_relationships(self):
        if self.person_emp_length is not None and self.person_emp_length > self.person_age:
            raise ValueError('Employment length cannot exceed age')
        if self.cb_person_cred_hist_length > self.person_age:
            raise ValueError('Credit history length cannot exceed age')
        if abs(self.loan_percent_income - self.loan_amnt / self.person_income) > .011:
            raise ValueError('loan_percent_income must match loan_amnt / person_income (rounded to 2 decimals)')
        return self

@app.get('/health')
def health():
    if not ml_model.get('models'):
        raise HTTPException(503, 'No trained models. Run python -m credit_risk.train first.')
    return {'status': 'ready', 'default_model': ml_model['manifest']['default_model']}

@app.get('/models')
def models():
    health()
    return {'default_model': ml_model['manifest']['default_model'], 'models': list(ml_model['models'])}

@app.post('/predict')
def predict(data: LoanApplication, model: str | None = None):
    health()
    name = model or ml_model['manifest']['default_model']
    if name not in ml_model['models']:
        raise HTTPException(404, f'Unknown model: {name}')
    bundle = ml_model['models'][name]
    frame = pd.DataFrame([data.model_dump()]).fillna(float('nan'))
    probability = float(bundle['model'].predict_proba(frame)[:, 1][0])
    threshold = float(bundle['threshold'])
    prediction = int(probability >= threshold)
    return {'model': name, 'default_probability': probability, 'default_prediction': prediction,
            'threshold': threshold, 'Result': 'High Risk' if prediction else 'Low Risk'}

@app.post('/explain')
def explain(data: LoanApplication, model: str | None = None):
    health()
    name = model or ml_model['manifest']['default_model']
    if name not in ml_model['models']:
        raise HTTPException(404, 'Unknown model')
    service = ml_model.get('explanation_service')
    if service is None:
        raise HTTPException(503, 'Explanations unavailable. Check the original CSV and model manifest.')
    frame = pd.DataFrame([data.model_dump()]).fillna(float('nan'))
    try:
        result = service.explain(ml_model['models'][name]['model'], frame)
    except ImportError as exc:
        raise HTTPException(503, 'SHAP dependency could not load. Predictions remain available. Windows Application Control may require administrator approval.') from exc
    except BlockingIOError as exc:
        raise HTTPException(429, str(exc)) from exc
    except Exception as exc:
        logging.exception('SHAP explanation failed')
        raise HTTPException(500, 'Explanation failed; prediction is still available. Check the server log.') from exc
    result['model'] = name
    result['threshold'] = float(ml_model['models'][name]['threshold'])
    return result

app.include_router(uci_router)

app.mount('/', StaticFiles(directory=ROOT/'static', html=True), name='static')
