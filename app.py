# flask api for big data project

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('bigdataproject')

from src.data.make_dataset import load_train_processed_data, load_test_processed_data
from src.visualization.visualize import count_target_value, age_of_client_at_time_application, age_of_client_deemed_capable_at_time_application, age_of_client_deemed_incapable_at_time_application, employment_at_time_application, employment_at_time_Application_deemed_capable, employment_at_time_Application_deemed_incapable
from src.models.predict_model import load_model, predict_from_json
from src.models.train_model import train_model_randomForest,train_model_randomForest_from_json

app = Flask(__name__)
model = load_model('random_forest.pkl')
df_train = load_train_processed_data()
df_test = load_test_processed_data()
df = pd.concat([df_train, df_test], axis=0)

@app.route('/api/visualize/count_target_value', methods=['GET'])
def count_target_value_api():
    return jsonify(count_target_value(df))

@app.route('/api/visualize/age_of_client_at_time_application', methods=['GET'])
def age_of_client_at_time_application_api():
    return jsonify(age_of_client_at_time_application(df))

@app.route('/api/visualize/age_of_client_deemed_capable_at_time_application', methods=['GET'])
def age_of_client_deemed_capable_at_time_application_api():
    return jsonify(age_of_client_deemed_capable_at_time_application(df))

@app.route('/api/visualize/age_of_client_deemed_incapable_at_time_application', methods=['GET'])
def age_of_client_deemed_incapable_at_time_application_api(df):
    return jsonify(age_of_client_deemed_incapable_at_time_application())

@app.route('/api/visualize/employment_at_time_application', methods=['GET'])
def employment_at_time_application_api():
    return jsonify(employment_at_time_application(df))

@app.route('/api/visualize/employment_at_time_Application_deemed_capable', methods=['GET'])
def employment_at_time_Application_deemed_capable_api():
    return jsonify(employment_at_time_Application_deemed_capable(df))

@app.route('/api/visualize/employment_at_time_Application_deemed_incapable', methods=['GET'])
def employment_at_time_Application_deemed_incapable_api():
    return jsonify(employment_at_time_Application_deemed_incapable(df))

@app.route('/api/models/decision_tree/predict', methods=['POST'])
def predict():
    # get data from request
    data = request.get_json()
    # predict
    prediction = predict_from_json(model, data)
    # return response
    return jsonify({'prediction': prediction})

@app.route('/api/models/decision_tree/train', methods=['POST'])
def train_randomForest():
    # get data from request
    data = request.get_json()

    try:
        model_name = data['model_name']
    except:
        pass

    try:
        mlflow_experiment_name = data['mlflow_experiment_name']
    except:
        pass

    # train model
    try:
        train_model_randomForest_from_json(df_train,data)
        # return response
        return jsonify({'message': 'Model Trained', 'status': 'success', 'model name': model_name,'mlflow_experiment_name':mlflow_experiment_name})
    except Exception as e:
        return jsonify({'message': 'Model Not Trained', 'status': 'failed', 'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port=8081)
