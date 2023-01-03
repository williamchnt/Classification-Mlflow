# flask api for big data project

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('bigdataproject')

from src.data.make_dataset import load_processed_data
from src.visualization.visualize import count_target_value, age_of_client_at_time_application, age_of_client_deemed_capable_at_time_application, age_of_client_deemed_incapable_at_time_application, employment_at_time_application, employment_at_time_Application_deemed_capable, employment_at_time_Application_deemed_incapable
from src.models.predict_model import load_model, predict_from_json
from src.models.train_model import train_model_randomForest

app = Flask(__name__)
model = load_model('random_forest.pkl')
df = load_processed_data()

@app.route('/api', methods=['GET'])
def api():
    return jsonify({'message': 'Welcome to the API'})

@app.route('/api/visualize', methods=['GET'])
def visualize():
    return jsonify({'message': 'Welcome to the Visualization API'})

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

@app.route('/api/models', methods=['GET'])
def models():
    return jsonify({'message': 'Welcome to the Models API'})

@app.route('/api/models/decision_tree', methods=['GET'])
def decision_tree():
    return jsonify({'message': 'Welcome to the Decision Tree API'})

@app.route('/api/models/decision_tree/predict', methods=['POST'])
def predict():
    # get data from request
    data = request.get_json()
    # predict
    prediction = predict_from_json(model, data)
    # return response
    return jsonify({'prediction': prediction})

@app.route('/api/models/decision_tree/train', methods=['GET'])
def train_randomForest(n_estimators, max_depth, max_features, min_samples_leaf, min_samples_split, random_state=42,test_size=0.2, save_model=True, model_name='model.pkl',record_model=True, mlflow_experiment_name='random_forest'):
    # get data from request
    # data = request.get_json()
    # train model
    train_model_randomForest(df,n_estimators, max_depth, max_features, min_samples_leaf, min_samples_split, random_state,test_size, save_model, model_name,record_model, mlflow_experiment_name)
    # return response
    return jsonify({'message': 'Model Trained'})

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port=8081)
