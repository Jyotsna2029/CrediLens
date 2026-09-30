from pathlib import Path
import joblib
import numpy as np
from fastapi.testclient import TestClient
from main import app
from credit_risk.persistence import load_bundle
from credit_risk.data import load_data, FEATURES

PAYLOAD = dict(person_age=30, person_income=60000, person_home_ownership='RENT', person_emp_length=5,
    loan_intent='PERSONAL', loan_grade='B', loan_amnt=12000, loan_int_rate=10,
    loan_percent_income=.2, cb_person_default_on_file='N', cb_person_cred_hist_length=7)


def test_native_xgboost_matches_existing_predictions():
    directory=Path('artifacts')
    old=joblib.load(directory/'xgboost.joblib')
    new=load_bundle(directory,'xgboost')
    rows=load_data('credit_risk_dataset.csv')[FEATURES].head(50)
    np.testing.assert_allclose(old['model'].predict_proba(rows),new['model'].predict_proba(rows),atol=1e-7)
    assert old['threshold']==new['threshold']


def test_all_five_explanations_match_served_probability():
    with TestClient(app) as client:
        for name in client.get('/models').json()['models']:
            pred=client.post('/predict',params={'model':name},json=PAYLOAD).json()
            result=client.post('/explain',params={'model':name},json=PAYLOAD)
            assert result.status_code==200, result.text
            data=result.json()
            assert len(data['features'])==11
            assert abs(data['default_probability']-pred['default_probability'])<1e-7
            assert abs(data['base_probability']+sum(f['contribution'] for f in data['features'])-pred['default_probability'])<1e-5
        assert client.post('/explain',params={'model':'unknown'},json=PAYLOAD).status_code==404
        assert client.post('/explain',json={**PAYLOAD,'person_income':0}).status_code==422


def test_missing_value_and_out_of_range_explanation():
    with TestClient(app) as client:
        sample={**PAYLOAD,'loan_amnt':1000000,'person_income':6000000,'loan_percent_income':.17,'person_emp_length':None}
        result=client.post('/explain',params={'model':'xgboost'},json=sample)
        assert result.status_code==200, result.text
        data=result.json()
        assert any('Loan amount' in message for message in data['warnings'])
        assert next(f for f in data['features'] if f['feature']=='person_emp_length')['value'] is None
