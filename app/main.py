from typing import List

from fastapi import FastAPI
import joblib
import numpy as np
import xgboost as xgb
from pydantic import BaseModel
import os.path


model_path = (os.path.dirname(__file__) + '/../learning/bst_model.pkl')
model = joblib.load(model_path)


def make_prediction(model, payload):
    features = preprocess_payload(payload)

    predictions = model.predict(features)
    return predictions[0].tolist()


def preprocess_payload(payload):
    array = np.asarray(payload.instance).reshape(1, -1)
    xgb_features = xgb.DMatrix(array)

    return xgb_features


class ModelPayload(BaseModel):
    instance: List[float]


app = FastAPI()


@app.post("/predictions")
def dispatch_predictions(payload: ModelPayload):
    return {"predictions": make_prediction(model, payload)}
