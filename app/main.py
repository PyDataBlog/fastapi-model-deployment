# Import Needed Libraries
import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from data.predictions_handler import get_predictions
from loguru import logger

# Initiate app instance
app = FastAPI(title='Forged Or Not Forged', version='1.0',
              description='A simple Neural Network that classifies a banknote as either forged or not')


# Design the incoming feature data
class Features(BaseModel):
    variance_of_wavelet: float
    skewness_of_wavelet: float
    curtosis_of_wavelet: float
    entropy_of_wavelet: float

# Initiate logging
log_format = ""
logger.add(sink='/data/log_files/log.log', format=log_format, level='DEBUG')


# Api root/home
@app.get('/')
@app.get('/home')
def read_home():
    return {'message': 'Fake or Not API live!'}


# Prediction endpoint
@app.post('/predict')
def get_prediction(incoming_data: Features):
    # retrieve incoming json data as a dictionary
    new_data = incoming_data.dict()

    # Make predictions based on the incoming data and saved neural net
    preds = get_predictions(new_data)

    # Return the predicted class and the predicted probability
    return {'predicted_class': round(float(preds.flatten())), 'predicted_prob': float(preds.flatten())}


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")


