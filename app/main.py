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
log_format = "{time} | {level} | {message} | {file} | {line} | {function} | {exception}"
logger.add(sink='app/data/log_files/logs.log', format=log_format, level='DEBUG', compression='zip')


# Api root or home endpoint
@app.get('/')
@app.get('/home')
def read_home():
    """
    Home endpoint which can be used to test the availability of the application.

    :return: Dict with key 'message' and value 'Fake or Not API live!'
    """
    logger.debug('User checked the root page')
    return {'message': 'Fake or Not API live!'}


# Prediction endpoint
@app.post('/predict')
@logger.catch()  # catch any unexpected breaks
def get_prediction(incoming_data: Features):
    """
    This endpoint serves the predictions based on the values received from a user and the saved model.

    :param incoming_data: JSON with keys representing features and values representing the associated values.
    :return: Dict with keys 'predicted_class' (class predicted) and 'predicted_prob' (probability of prediction)
    """
    # retrieve incoming json data as a dictionary
    new_data = incoming_data.dict()
    logger.info('User sent some data for predictions')

    # Make predictions based on the incoming data and saved neural net
    preds = get_predictions(new_data)
    logger.debug('Predictions successfully generated for the user')

    # Return the predicted class and the predicted probability
    return {'predicted_class': round(float(preds.flatten())), 'predicted_prob': float(preds.flatten())}


if __name__ == "__main__":
    # Run app with uvicorn with port and host specified. Host needed for docker port mapping
    uvicorn.run(app, port=8000, host="0.0.0.0")


