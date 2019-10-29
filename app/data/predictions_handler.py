import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import os
from loguru import logger


@logger.catch
def get_predictions(data):
    """
    A function that reshapes the incoming JSON data, loads the saved model objects
    and returns predicted class and probability.

    :param data: Dict with keys representing features and values representing the associated value
    :return: Dict with keys 'predicted_class' (class predicted) and 'predicted_prob' (probability of prediction)
    """
    # Convert new data dict as a DataFrame and reshape the columns to suit the model
    new_data = {k: [v] for k, v in data.items()}

    new_data_df = pd.DataFrame.from_dict(new_data)

    new_data_df = new_data_df[['variance_of_wavelet', 'skewness_of_wavelet',
                               'curtosis_of_wavelet', 'entropy_of_wavelet']]

    # Load saved standardising object
    scaler = joblib.load('app/data/scaler.joblib')
    logger.debug('Saved standardising object successfully loaded')

    # Load saved keras model
    model = tf.keras.models.load_model('app/data/banknote_authentication_model.h5')
    logger.debug('Saved ANN model loaded successfully')

    # Scale new data using the loaded object
    X = scaler.transform(new_data_df.values)
    logger.debug('Incoming data successfully standardised with saved object')

    # Make new predictions from the newly scaled data and return this prediction
    preds = model.predict(X)

    return preds