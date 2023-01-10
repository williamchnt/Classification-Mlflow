# load model
import joblib
import numpy as np
import pandas as pd

def load_model(model_name):
    model = joblib.load('bigdataproject/models/'+model_name+".pkl")
    return model

# predict
def predict(model, X):
    y_pred = model.predict(X)[0]
    return y_pred

def predict_from_json(model, json_data):
    data = np.array(list(json_data.values())).reshape(1, -1)
    return predict(model, data)

def predict_from_json_model_name(json_data,model_name='random_forest'):
    model = load_model(model_name)
    return predict_from_json(model, json_data)

def predict_from_json_model_name_test(rowNb=0,model_name='random_forest'):
    model = load_model(model_name)
    df_test = pd.read_csv('bigdataproject/data/processed/application_test.csv')

    # Keep row
    df_test = df_test.iloc[[rowNb]]

    y_pred = predict(model, df_test)
    return y_pred