import requests, json, asyncio
from flask import Flask, flash, jsonify, request, redirect, url_for, render_template
from model import features
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle

app = Flask(__name__)
app.secret_key = 'secret'

@app.route('/',methods=['POST','GET'])
def home():
	feature_list = features
	return render_template('index.html',features=feature_list)

@app.route('/predict',methods=['POST','GET'])
def predict():
	try:
		prediction_features = request.form

		# convert tuples to dictionary
		nums = '1234567890'
		prediction_dict = prediction_features.to_dict(flat=False)
		for item in prediction_dict:
			# if feature is not numerical
			if prediction_dict[item][0][0] not in nums:
				prediction_dict[item] = [prediction_dict[item][0]]
			# if feature should be an integer
			else:
				prediction_dict[item] = [int(prediction_dict[item][0])]

		# convert predictionDict to dataframe
		df = pd.DataFrame.from_dict(prediction_dict)

		# label encode the non-numerical features
		for col in df:
			if df[col].dtypes == 'object':
				encoder = LabelEncoder()
				encoder.fit(df[col])
				encoder_values = encoder.transform(df[col])
				df[col] = encoder_values
		
		# load the model
		loaded_model = pickle.load(open('rf-bank-trained.sav','rb'))
		# make prediction
		result = loaded_model.predict(df)
		if result[0] == 0:
			print('Will NOT subscribe')
			flash('Will NOT subscribe','message')
		else:
			print('Will subscribe')
			flash('Will subscribe','message')

	except:
		pass
	return redirect(url_for("home"))

if __name__ == "__main__":
	#app.debug = True
	app.run()