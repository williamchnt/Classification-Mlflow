# import libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
import mlflow
import sys

def train_model_randomForest_from_json(df_train, json_data, model_name='random_forest.pkl',record_model=True, mlflow_experiment_name='random_forest'):
    # Get data from json
    # df_train = pd.read_json(json_data['df_train'])
    # df_test = pd.read_json(json_data['df_test'])
    n_estimators = json_data['n_estimators']
    max_depth = json_data['max_depth']
    max_features = json_data['max_features']
    min_samples_leaf = json_data['min_samples_leaf']
    min_samples_split = json_data['min_samples_split']
    random_state = json_data['random_state']
    save_model = json_data['save_model']
    model_name = json_data['model_name']
    record_model = json_data['record_model']
    mlflow_experiment_name = json_data['mlflow_experiment_name']
    test_size = json_data['test_size']
    random_state_split = json_data['random_state_split']


    # Split data
    X_train = df_train.drop('TARGET', axis=1)
    y_train = df_train['TARGET']
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state_split)


    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, random_state=random_state)
    
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # F1 score
    f1 = f1_score(y_test, y_pred)

    # ROC AUC score
    roc_auc = roc_auc_score(y_test, y_pred)

    # Record model
    if record_model:
        mlflow.set_experiment(mlflow_experiment_name)
        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_param('max_depth', max_depth)
        mlflow.log_param('max_features', max_features)
        mlflow.log_param('min_samples_leaf', min_samples_leaf)
        mlflow.log_param('min_samples_split', min_samples_split)
        mlflow.log_param('random_state', random_state)
        mlflow.log_param('test_size', test_size)
        mlflow.log_param('random_state_split', random_state_split)
        mlflow.log_metric('accuracy', acc)
        mlflow.log_metric('f1', f1)
        mlflow.log_metric('roc_auc', roc_auc)
        mlflow.sklearn.log_model(model, 'model')

    # Save model
    if save_model:
        joblib.dump(model, "bigdataproject/models/"+model_name+".pkl")

    return model


def train_model_randomForest(df_train,df_test,n_estimators, max_depth, max_features, min_samples_leaf, min_samples_split, random_state=42,test_size=0.2, save_model=True, model_name='random_forest.pkl',record_model=True, mlflow_experiment_name='random_forest'):
    # Split data
    X_train = df_train.drop('TARGET', axis=1)
    X_test = df_test.drop('TARGET', axis=1)
    y_train = df_train['TARGET']
    y_test = df_test['TARGET']

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, random_state=random_state)
    
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # F1 score
    f1 = f1_score(y_test, y_pred)

    # ROC AUC score
    roc_value = roc_auc_score(y_test, y_pred)

    if record_model:
        # Record model in mlflow
        # new experiment in mlflow
        mlflow.set_experiment(mlflow_experiment_name)

        # log the model
        with mlflow.start_run():
            mlflow.log_param('n_estimators', n_estimators)
            mlflow.log_param('max_depth', max_depth)
            mlflow.log_param('max_features', max_features)
            mlflow.log_param('min_samples_leaf', min_samples_leaf)
            mlflow.log_param('min_samples_split', min_samples_split)
            mlflow.log_metric('roc_auc', roc_value)
            mlflow.log_metric('Accuracy', acc)
            mlflow.log_metric('f1', f1)
            mlflow.sklearn.log_model(model, model_name)

    if save_model:
        # Save model
        joblib.dump(model, "bigdataproject/models"+model_name+".pkl")