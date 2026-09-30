import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from credit_risk.evaluation import choose_threshold
from credit_risk.data import load_data, clean_data, split_data, FEATURES
from credit_risk.models import make_model


def test_threshold_maximizes_actual_f1():
    y=np.array([0,1,0,1,1,0]); p=np.array([.1,.2,.3,.4,.8,.9])
    t=choose_threshold(y,p)
    assert f1_score(y,p>=t) == max(f1_score(y,p>=x) for x in p)


def test_cleaning_keeps_missing_employment():
    df=load_data('credit_risk_dataset.csv').head(10).copy()
    df.loc[df.index[0],'person_emp_length']=np.nan
    assert df.index[0] in clean_data(df).index


def test_splits_are_disjoint():
    parts=split_data(clean_data(load_data('credit_risk_dataset.csv')))
    a,b,c=map(lambda x:set(x.index),parts)
    assert not (a&b or a&c or b&c)


def test_preprocessing_handles_missing_and_unknown_category():
    df=clean_data(load_data('credit_risk_dataset.csv')).sample(300,random_state=5)
    model=make_model('logistic_regression').fit(df[FEATURES],df.loan_status)
    row=df[FEATURES].head(1).copy()
    row['person_income']=np.nan
    row['person_home_ownership']='UNSEEN'
    assert np.isfinite(model.predict_proba(row)).all()


def test_api_all_trained_models_and_validation():
    from fastapi.testclient import TestClient
    from main import app
    payload=dict(person_age=30,person_income=60000,person_home_ownership='RENT',
        person_emp_length=5,loan_intent='PERSONAL',loan_grade='B',loan_amnt=12000,
        loan_int_rate=10,loan_percent_income=.2,cb_person_default_on_file='N',cb_person_cred_hist_length=7)
    with TestClient(app) as client:
        assert client.get('/health').status_code==200
        names=client.get('/models').json()['models']
        assert len(names)==5
        for name in names:
            result=client.post('/predict',params={'model':name},json=payload)
            assert result.status_code==200
            data=result.json()
            assert data['model']==name and 0<=data['default_probability']<=1
            assert data['default_prediction']==int(data['default_probability']>=data['threshold'])
        assert client.post('/predict',json=payload).status_code==200
        assert client.post('/predict',params={'model':'bad'},json=payload).status_code==404
        assert client.post('/predict',json={**payload,'person_income':0}).status_code==422
        assert client.post('/predict',json={**payload,'loan_percent_income':.9}).status_code==422
