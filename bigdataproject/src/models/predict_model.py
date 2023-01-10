# load model
import joblib
import numpy as np

def load_model(model_name):
    model = joblib.load('bigdataproject/models/'+model_name)
    return model

# predict
def predict(model, X):
    y_pred = model.predict(X)[0]
    return y_pred

def predict_from_json(model, json_data):
    data = np.array(list(json_data.values())).reshape(1, -1)
    return predict(model, data)