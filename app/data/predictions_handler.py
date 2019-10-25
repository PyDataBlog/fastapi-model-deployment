import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import os


def get_predictions(data):
    print(os.getcwd())
    new_data = {k:[v] for k, v in data.items()}

    new_data_df = pd.DataFrame.from_dict(new_data)
    new_data_df = new_data_df[['variance_of_wavelet', 'skewness_of_wavelet',
                               'curtosis_of_wavelet', 'entropy_of_wavelet']]

    scaler = joblib.load('app/data/scaler.joblib')
    model = tf.keras.models.load_model('app/data/banknote_authentication_model.h5')

    X = scaler.transform(new_data_df.values)
    preds = model.predict(X)

    return preds