# Import Needed Libraries
import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from data.predictions_handler import get_predictions


# Initiate app instance
app = FastAPI(title='Forged Or Not Forged', version='1.0',
              description='A simple Neural Network that classifies a banknote as either forged or not')


# Design the incoming feature data
class Features(BaseModel):
    variance_of_wavelet: float
    skewness_of_wavelet: float
    curtosis_of_wavelet: float
    entropy_of_wavelet: float

# Api root/home
@app.get('/')
@app.get('/home')
def read_home():
    return {'message': 'Fake or Not API live!'}


# Prediction endpoint
@app.post('/predict')
def get_prediction(incoming_data: Features):
    new_data = incoming_data.dict()

    preds = get_predictions(new_data)

    return {'predicted_class': round(float(preds.flatten())), 'predicted_prob': float(preds.flatten())}


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")


