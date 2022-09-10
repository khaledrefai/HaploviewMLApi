import json
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score , mean_absolute_error
import datetime
from sklearn.svm import LinearSVR
from sklearn.linear_model import BayesianRidge
from flask import Flask, jsonify,request
import numpy as np
import logging
import boto3
import os

app = Flask(__name__)

pcrAA = make_pipeline(StandardScaler(), PCA(n_components=500), LinearRegression())
pcrAB = make_pipeline(StandardScaler(), PCA(n_components=200), LinearRegression())
pcrBA = make_pipeline(StandardScaler(), PCA(n_components=200), LinearRegression())
pcrBB = make_pipeline(StandardScaler(), PCA(n_components=200), LinearRegression())
pcrunknownDH  = make_pipeline(StandardScaler(), PCA(n_components=200), LinearRegression())


pcrAA = load('./saved_models/pcrAA.joblib')
pcrAB = load('./saved_models/pcrAB.joblib') 
pcrBA = load('./saved_models/pcrBA.joblib') 
pcrBB = load('./saved_models/pcrBB.joblib') 
pcrunknownDH = load('saved_models/pcrunknownDH.joblib') 

@app.route('/predict', methods=['POST'])
def predict():
     data = request.json
     pos1 =  np.array(data['pos1'])
     other = data['other']
     otherArr = np.empty([1200,1])
     retList = []
     for snps in other :
        snpsArr = np.array(snps)
        otherArr = np.concatenate((pos1, snpsArr), axis=0).reshape(1, -1)
        retValue = {'aa': str(int(pcrAA.predict(otherArr)[0])),'ab':str(int(pcrAB.predict(otherArr)[0])),
        'ba': str(int(pcrBA.predict(otherArr)[0]))
                 ,'bb': str(int(pcrBB.predict(otherArr)[0])),'unknowen': str(int(pcrunknownDH .predict(otherArr)[0])) }
        retList.append(retValue)         
     print(len(retList))
     return jsonify( retList) 


if __name__ == '__main__':
    app.run(debug=True, port=5000)