#import libraries
import pandas as pd

def load_processed_data():
    # Load data
    df_train = pd.read_csv('../data/processed/application_train.csv')
    df_test = pd.read_csv('../data/processed/application_test.csv')
    df = pd.concat([df_train, df_test], axis=0)
    return df

# Dummy string columns
def dummy_string_columns(df_train, df_test):
    df_train = pd.get_dummies(df_train)
    df_test = pd.get_dummies(df_test)
    return df_train, df_test