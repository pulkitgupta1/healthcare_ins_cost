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

		# Reshape the Data as a Sample not Individual Features
		ex1 = np.array(sample_data).reshape(1,-1)
#		print(ex1)

		# ex1 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)
		model = joblib.load('savedmomdel.pkl')
		result_prediction=model.predict(ex1)[0]
		content=str(result_prediction)
		ur='https://www.google.com'
		content2='Hello sir.The predicted charges are '+content+' .Also you can see the dynamic dashboard of all the vital analytics at this url:'+ur
		msg=MIMEMultipart()
		msg['From']='yjain893.yj@gmail.com'
		msg['To']='yjain893.yj@gmail.com'
		msg['Subject']='Your prediction is ready'
		msg.attach(MIMEText(content2,'plain'))
        
		server=smtplib.SMTP('smtp.gmail.com',587)
		server.ehlo()
		server.starttls()
		server.login('yjain893.yj@gmail.com','8013815470yash')
		text=msg.as_string()
        
		server.sendmail('yjain893.yj@gmail.com','yjain893.yj@gmail.com',text)
		server.quit()
		# Reloading the Model
#		if model_choice == 'logitmodel':
#		    logit_model = joblib.load('saved.pkl')
#		    result_prediction = logit_model.predict(ex1)
#		elif model_choice == 'knnmodel':
#			knn_model = joblib.load('data/knn_model_iris.pkl')
#			result_prediction = knn_model.predict(ex1)
#		elif model_choice == 'svmmodel':
#			knn_model = joblib.load('data/svm_model_iris.pkl')
#			result_prediction = knn_model.predict(ex1)

	return render_template('index.html', children=children,
		region=region,
		sex=sex,
		age=age,
		bmi=bmi,
		smoker = smoker,
		result_prediction=result_prediction,
		)


if __name__ == '__main__':
	app.run(debug=True)