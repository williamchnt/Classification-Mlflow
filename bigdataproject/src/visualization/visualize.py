import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('../..')
from bigdataproject.src.data import load_data

def count_target_value():
    data = load_data.load_processed_data()
    data['TARGET'].value_counts().plot.pie(autopct='%1.1f%%')

def age_of_client_at_time_application():
    data = load_data.load_processed_data()
    plt.hist(data['DAYS_BIRTH'].values/-365, bins=10, edgecolor='black', color='#80489C')
    plt.title('Age of Client (in years) at the time of Application')
    plt.xlabel('Age Bucket')
    plt.ylabel('Number of Clients')
    plt.show()

def age_of_client_deemed_capable_at_time_application():
    df = load_data.load_processed_data()
    capable_days_birth = df[df['TARGET'] == 0]['DAYS_BIRTH'].values/-365
    plt.figure(figsize=(10,3))
    plt.hist(capable_days_birth, bins=10, edgecolor='black', color='#FF8FB1')
    plt.title('Age of Client (in years) deemed Capable at the time of Application')
    plt.xlabel('Age Bucket')
    plt.ylabel('Number of Clients')
    plt.show()

def age_of_client_deemed_incapable_at_time_application():
    df = load_data.load_processed_data()
    not_capable_days_birth = df[df['TARGET'] == 1]['DAYS_BIRTH'].values/-365
    plt.figure(figsize=(10,3))
    plt.hist(not_capable_days_birth, bins=10, edgecolor='black', color='#FCE2DB')
    plt.title('Age of Client (in years) deemed Not Capable at the time of Application')
    plt.xlabel('Age Bucket')
    plt.ylabel('Number of Clients')
    plt.show()

def employment_at_time_application():
    df = load_data.load_processed_data()
    plt.figure(figsize=(10,3))
    plt.hist(df['DAYS_EMPLOYED'].values / 365, bins=10, edgecolor='#80489C')
    plt.title('Employment (in years) at the time of Application')
    plt.xlabel('Employment years bucket')
    plt.ylabel('Number of Clients')
    plt.show()

def employment_at_time_Application_deemed_capable():
    df = load_data.load_processed_data()
    capable_days_employed = df[df['TARGET']==0]['DAYS_EMPLOYED'].values / 365
    plt.figure(figsize=(10,3))
    plt.hist(capable_days_employed, bins=10, edgecolor='black', color='#FF8FB1')
    plt.title('Employment (in years) deemed Capable at the time of Application')
    plt.xlabel('Employment years bucket')
    plt.ylabel('Number of Clients')
    plt.show()

def employment_at_time_Application_deemed_incapable():
    df = load_data.load_processed_data()
    not_capable_days_employed = df[df['TARGET']==1]['DAYS_EMPLOYED'].values / 365
    plt.figure(figsize=(10,3))
    plt.hist(not_capable_days_employed, bins=10, edgecolor='black', color='#FCE2DB')
    plt.title('Employment (in years) deemed Not Capable at the time of Application')
    plt.xlabel('Employment years bucket')
    plt.ylabel('Number of Clients')
    plt.show()

