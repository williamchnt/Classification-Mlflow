# import libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
import mlflow
import sys

def train_model_randomForest(df,n_estimators, max_depth, max_features, min_samples_leaf, min_samples_split, random_state=42,test_size=0.2, save_model=True, model_name='model.pkl',record_model=True, mlflow_experiment_name='random_forest'):
    # Split data
    X = df.drop('TARGET', axis=1)
    y = df['TARGET']
    model = RandomForestClassifier()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
 

    # Train model
    
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
            mlflow.log_metric('Accuracy', acc)
            mlflow.sklearn.log_model(model, model_name)

    if save_model:
        # Save model
        joblib.dump(model, model_name)