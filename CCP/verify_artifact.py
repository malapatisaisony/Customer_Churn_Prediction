import joblib
import pandas as pd

art = joblib.load('churn_model.joblib')
print('model_name:', art.get('model_name'))
model = art['model']

try:
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    sample = df.head(5).copy()
except Exception as e:
    sample = None

if sample is not None:
    if 'Churn' in sample.columns:
        sample_X = sample.drop(columns=['Churn'])
    else:
        sample_X = sample
    if 'customerID' in sample_X.columns:
        sample_X = sample_X.drop(columns=['customerID'])
    if 'TotalCharges' in sample_X.columns:
        sample_X['TotalCharges'] = pd.to_numeric(sample_X['TotalCharges'], errors='coerce')
    probs = model.predict_proba(sample_X)[:,1]
    print('predict_proba:', probs)
else:
    print('No sample available to verify')
