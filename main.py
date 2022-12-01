# import
from typing import Union
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI

classifier = joblib.load('classifier.joblib')
data = joblib.load('data.joblib')
threshold = joblib.load('threshold.joblib')
# imputer = joblib.load('imputer.joblib')
shap.initjs()

app = FastAPI() 



@app.get("/")
def read_root():
    return {"Hello": "W"}



@app.get("/predict")
def predic_loan(SK_ID_CURR:int):
    # SK_ID_CURR to X
    dt = data.loc[data['SK_ID_CURR'] == SK_ID_CURR,:]
    # prediction
    X = dt.drop(columns = ['TARGET','SK_ID_CURR'])
    predicted_proba = classifier.predict_proba(X)[:,1][0]
    if predicted_proba >= threshold:
        predicted_proba =  'credit refused, and the risk pencentage is ', predicted_proba*100
    else:
        predicted_proba = 'credit accepted, the loan will be repaid', predicted_proba*100
    return {'prediction': predicted_proba}

@app.get("/shapplot")
def predic_loan(SK_ID_CURR:int):
    # SK_ID_CURR to X
    dt = data.loc[data['SK_ID_CURR'] == SK_ID_CURR,:]
    X = dt.drop(columns = ['TARGET','SK_ID_CURR'])
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X)
    force_plot = shap.force_plot(explainer.expected_value[0], shap_values[0], X.values, feature_names = X.columns, matplotlib=False)
    
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"

    return {'force_plot_html': shap_html}