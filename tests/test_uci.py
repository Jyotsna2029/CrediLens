import json
from pathlib import Path
import pandas as pd
from credit_risk.uci_data import load_data,clean_data,split_data,FEATURES


def test_uci_target_and_id_exclusion():
    df=load_data('data/uci/credit_card_clients.csv')
    assert len(df)==30000 and len(FEATURES)==23
    assert 'ID' not in FEATURES and 'loan_status' not in FEATURES
    assert int(df.loan_status.sum())==6636


def test_identical_profiles_never_cross_splits():
    parts=split_data(clean_data(load_data('data/uci/credit_card_clients.csv')))
    groups=[set(pd.util.hash_pandas_object(x[FEATURES],index=False)) for x in parts]
    assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])


def test_uci_api_and_original_coexist():
    from fastapi.testclient import TestClient
    from main import app
    with TestClient(app) as client:
        assert client.get('/health').status_code==200
        assert client.get('/uci').status_code==200
        names=client.get('/uci/models').json()['models']
        assert len(names)==5
        sample=client.get('/uci/form-schema').json()['example']
        for model in names:
            for point in ['standard','f1']:
                r=client.post('/uci/predict',params={'model':model,'operating_point':point},json=sample)
                assert r.status_code==200, r.text
                assert 0<=r.json()['default_probability']<=1
                if point=='standard': assert r.json()['threshold']==.5
        assert client.post('/uci/predict',json={**sample,'ID':1}).status_code==422
        assert client.post('/uci/predict',json={**sample,'SEX':100}).status_code==422
        assert client.post('/uci/predict',json={**sample,'LIMIT_BAL':-1}).status_code==422
