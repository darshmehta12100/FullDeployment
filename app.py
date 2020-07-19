from flask import Flask,render_template,request
from sklearn import externals
import joblib
import pickle
import pandas as pd
import numpy as np
import requests

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        pass
    
    else:
        try:
            NewYork = float(request.form['NewYork'])
            California = float(request.form['California'])
            Florida = float(request.form['Florida'])
            RnDSpend = float(request.form['RnDSpend'])
            AdminSpend = float(request.form['AdminSpend'])
            MarketSpend = float(request.form['MarketSpend'])

            pred_args = [RnDSpend,AdminSpend,MarketSpend,California,Florida,NewYork]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1,-1)

            mul_reg = open('multiple_linear_model.pkl',"rb")
            ml_model = joblib.load(mul_reg)

            model_prediction = ml_model.predict(pred_args_arr)
            model_prediction = round(float(model_prediction),2)
        
        except ValueError:
            return 'Please Check if all the values entered are proper'

        return render_template('predict.html',prediction = model_prediction)


if __name__ == "__main__":
    app.run(host='0.0.0.0')