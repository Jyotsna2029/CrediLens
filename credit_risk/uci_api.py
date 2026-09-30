"""UCI-specific API and a simple form. Original loan form remains separate."""
from pathlib import Path
import json
import os
from .persistence import load_bundle
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import ConfigDict, Field, create_model
from .uci_data import FEATURES, CATEGORICAL

router = APIRouter(prefix='/uci', tags=['Credit card default (UCI)'])
loaded = {}
fields = {}
for name in FEATURES:
    if name == 'LIMIT_BAL': definition = (float, Field(gt=0))
    elif name == 'AGE': definition = (int, Field(ge=18, le=100))
    elif name == 'SEX': definition = (int, Field(ge=1, le=2))
    elif name == 'EDUCATION': definition = (int, Field(ge=0, le=6))
    elif name == 'MARRIAGE': definition = (int, Field(ge=0, le=3))
    elif name.startswith('PAY_') and not name.startswith('PAY_AMT'): definition = (int, Field(ge=-2, le=9))
    elif name.startswith('PAY_AMT'): definition = (float, Field(ge=0))
    else: definition = (float, Field())  # Bill balances can be negative.
    fields[name] = definition
CreditCardApplication = create_model('CreditCardApplication', __config__=ConfigDict(extra='forbid', allow_inf_nan=False), **fields)


def load_models(root):
    loaded.clear()
    directory = Path(os.environ.get('UCI_ARTIFACTS', str(root/'artifacts_uci')))
    if (directory/'manifest.json').exists():
        manifest = json.loads((directory/'manifest.json').read_text())
        if manifest.get('dataset') != 'uci':
            raise ValueError('UCI_ARTIFACTS must contain a UCI run')
        loaded['manifest'] = manifest
        loaded['models'] = {name: load_bundle(directory, name) for name in manifest['models']}


@router.get('/models')
def models():
    if not loaded.get('models'):
        raise HTTPException(503, 'Train with --dataset uci --output artifacts_uci first')
    return {'models': list(loaded['models']), 'default_model': loaded['manifest']['default_model']}


@router.post('/predict')
def predict(data: CreditCardApplication, model: str | None = None, operating_point: str = 'f1'):
    models()
    name = model or loaded['manifest']['default_model']
    if name not in loaded['models']:
        raise HTTPException(404, 'Unknown UCI model')
    if operating_point not in ['f1', 'standard']:
        raise HTTPException(422, 'operating_point must be f1 or standard')
    frame = pd.DataFrame([data.model_dump()])
    for col in CATEGORICAL:
        frame[col] = frame[col].astype(int).astype(str)
    bundle = loaded['models'][name]
    p = float(bundle['model'].predict_proba(frame)[:, 1][0])
    threshold = float(bundle['threshold']) if operating_point == 'f1' else .5
    return {'dataset': 'UCI credit card clients', 'model': name, 'default_probability': p,
            'threshold': threshold, 'default_prediction': int(p >= threshold),
            'Result': 'High Risk' if p >= threshold else 'Low Risk'}


@router.get('/form-schema')
def form_schema():
    example = {name: 0 for name in FEATURES}
    example.update(LIMIT_BAL=120000, AGE=30, SEX=2, EDUCATION=2, MARRIAGE=1)
    for i in range(1,7):
        example[f'BILL_AMT{i}']=10000; example[f'PAY_AMT{i}']=2000
    return {'example': example, 'schema': CreditCardApplication.model_json_schema()}


@router.get('', response_class=HTMLResponse)
def page():
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Credit Card Default</title>
<style>body{font:16px system-ui;max-width:900px;margin:40px auto;padding:20px;background:#f4f6fa;color:#15253b}form{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}label{display:flex;flex-direction:column;gap:5px}input,select,button{padding:10px}pre{white-space:pre-wrap;background:white;padding:20px}button{margin-top:20px}</style>
<h1>Credit card default prediction</h1><p>UCI historical benchmark. All amounts are in New Taiwan dollars. PAY_0 is September; PAY_2–PAY_6 are August–April. Preserve the original dataset codes.</p>
<p><a href="/">Loan application demo</a> · <a href="/docs">API documentation</a></p>
<label>Model<select id="model"></select></label><label>Threshold<select id="point"><option value="f1">Validation F1 threshold</option><option value="standard">Standard 0.5 threshold</option></select></label>
<p>Prefilled values are an illustrative example.</p><form id="form"></form><button id="submit">Predict default risk</button><pre id="result">Loading models…</pre>
<script>
const result=document.getElementById('result');
Promise.all([fetch('/uci/models').then(r=>{if(!r.ok)throw Error('Train UCI models first');return r.json()}),fetch('/uci/form-schema').then(r=>r.json())]).then(([m,s])=>{
 m.models.forEach(n=>{const o=document.createElement('option');o.value=n;o.textContent=n;o.selected=n===m.default_model;document.getElementById('model').append(o)});
 Object.entries(s.example).forEach(([name,value])=>{const label=document.createElement('label');label.textContent=name;const input=document.createElement('input');input.name=name;input.type='number';input.value=value;input.required=true;input.step=s.schema.properties[name].type==='integer'?'1':'any';const p=s.schema.properties[name];if(p.minimum!==undefined)input.min=p.minimum;if(p.maximum!==undefined)input.max=p.maximum;label.append(input);document.getElementById('form').append(label)});result.textContent='Ready';
}).catch(e=>result.textContent=e.message);
document.getElementById('submit').onclick=async()=>{const form=document.getElementById('form');if(!form.reportValidity())return;const data=Object.fromEntries([...new FormData(form)].map(([k,v])=>[k,Number(v)]));try{const r=await fetch('/uci/predict?model='+encodeURIComponent(document.getElementById('model').value)+'&operating_point='+document.getElementById('point').value,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});result.textContent=JSON.stringify(await r.json(),null,2)}catch(e){result.textContent=e.message}};
</script></html>'''
