'''
modified https://github.com/jaadeoye/opmd-mt-deepsurv-app
'''
from flask import Flask, render_template, url_for, request
import pandas as pd 
import numpy as np 
import pickle
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import torch
import torchtuples as tt
import celery
from pycox.models import CoxPH

app = Flask(__name__)
#app.debug=True
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def analyze():
    if request.method == 'POST':
        age = request.form['age']
        Sex = request.form['Sex']
        race = request.form['race']
        Primary_Site = request.form['Primary_Site']
        Lauren = request.form['Lauren']
        Grade = request.form['Grade']
        T_7th = request.form['T_7th']
        M_7th = request.form['M_7th']
        N_7th = request.form['N_7th']
        Surgery = request.form['Surgery']
        radiation = request.form['radiation']
        chemo = request.form['chemo']
        nodes_ex = request.form['nodes_ex']
        nodes_pos = request.form['nodes_pos']
        size = request.form['size']
        model_choice = request.form['model_choice']
        dataframe={"age":age,
                  "Sex":Sex, 
                  "race":race, 
                  "Primary_Site":Primary_Site,
                  "Lauren":Lauren,
                  "Grade":Grade,
                  "T_7th":T_7th,
                  "M_7th":M_7th,
                  "Surgery":Surgery,
                  "radiation":radiation,
                  "chemo":chemo,
                  "nodes_ex":nodes_ex,
                  "nodes_pos":nodes_pos,
                  "size":size}
        for k,v in dataframe.items():
            dataframe[k]=float(v)
        ex1=pd.DataFrame.from_dict(dataframe,orient='index').T
        age=[60,65,67,60,65,67,60,65,67]
        Sex=[1,1,1,1,1,0,0,0,0]
        race=[1,2,3,1,2,3,1,2,3]
        Primary_Site=[1,2,3,4,5,6,7,8,9]
        Lauren=[1,2,3,1,2,3,1,2,3]
        Grade=[1,2,3,4,1,2,3,1,2]
        T_7th=[1,2,3,4,5,6,1,2,3]
        M_7th=[0,0,0,0,0,1,1,1,1]
        Surgery=[1,1,1,1,1,2,2,2,2]
        radiation=[0,1,0,1,0,1,0,1,1]
        chemo=[0,1,0,1,0,1,0,1,1]
        nodes_ex=[10,20,30,40,50,60,80,70,10]
        nodes_pos=[10,10,22,23,45,67,89,12,45]
        size=[50,56,78,90,34,56,67,34,23]

        dataframe2={"age":age,
           "Sex":Sex, 
           "race":race, 
           "Primary_Site":Primary_Site,
           "Lauren":Lauren,
           "Grade":Grade,
           "T_7th":T_7th,
           "M_7th":M_7th,
           "Surgery":Surgery,
           "radiation":radiation,
           "chemo":chemo,
           "nodes_ex":nodes_ex,
           "nodes_pos":nodes_pos,
           "size":size}
        ex2=pd.DataFrame.from_dict(dataframe2,orient='index').T
        ex3=pd.concat([ex2,ex1])
        if model_choice == 'deepsurv':
            fname_model = "static/leoss_DeepSurv_CV_death_CV.pkl.gzip"
            deepsurv_model= joblib.load(fname_model)
            enc = deepsurv_model[0][1]
            model = deepsurv_model[1][1]
            surv = model.predict_surv_df(enc.fit_transform(ex3).astype("float32"))
            plt.plot(surv.iloc[:,9], color='red', linewidth=2)
            plt.xlabel("Duration in months")
            plt.ylabel("GC-Specific motility-free probability")
            plt.savefig('static/dps.jpg',  bbox_inches='tight')

        return render_template('predict.html', result_prediction = 'Prediction plot', url = 'static/dps.jpg', model_selected=model_choice)
       

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=9030)
