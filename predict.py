import pickle
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

SYMPTOMS = ['discomfort_eye_strain', 'redness_in_eye', 'itchiness_irritation_in_eye']

app = FastAPI(title='dry-eye-triage')

with open('model.bin', 'rb') as f_in:
    model = pickle.load(f_in)


class Patient(BaseModel):
    model_config = ConfigDict(extra='forbid')
    discomfort_eye_strain: Literal[0, 1]
    redness_in_eye: Literal[0, 1]
    itchiness_irritation_in_eye: Literal[0, 1]


class Prediction(BaseModel):
    dry_eye_probability: float
    dry_eye: bool


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict')
def predict(patient: Patient) -> Prediction:
    data = patient.model_dump()
    X = pd.DataFrame([{'any_symptom': int(sum(data[c] for c in SYMPTOMS) > 0)}])
    prob = float(model.predict_proba(X)[0, 1])
    return Prediction(dry_eye_probability=prob, dry_eye=prob >= 0.5)
