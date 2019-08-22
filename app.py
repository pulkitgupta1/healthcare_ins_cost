from flask import Flask,render_template,url_for,request
from flask_material import Material

# EDA PKg
import pandas as pd 
import numpy as np 
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# ML Pkg
from sklearn.externals import joblib


app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/preview')
def preview():
    df = pd.read_csv("data/insurance.csv")
    return render_template("preview.html",df_view = df)

@app.route('/',methods=["POST"])
def analyze():
	if request.method == 'POST':
		age = float(request.form['age'])
		sex = request.form['sex']
		bmi = float(request.form['bmi'])
		children = float(request.form['children'])
		smoker = request.form['smoker']
		region = request.form['region']
#		charges = request.form['model_choice']
		lbsex = joblib.load('labelsex.pkl')
		lbsmoker = joblib.load('labelsmoker.pkl')
		lbregion = joblib.load('labelregion.pkl')
        
		sex= lbsex.transform([sex])[0]
		region = lbregion.transform([region])[0]
		smoker = lbsmoker.transform([smoker])[0]
        
#		print(sex)
		# Clean the data by convert from unicode to float 
    
        
		sample_data = [age,sex,bmi,children,smoker,region]
        
		print(sample_data)
#		clean_data = [float(i) for i in sample_data]

