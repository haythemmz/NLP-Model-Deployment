from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pandas as pd 
import pickle 
import re 
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
import nltk 
from nltk.corpus import stopwords



app = Flask(__name__)

@app.route('/')


@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	

	if request.method == 'POST':
		message = request.form['message']
		bow= pickle.load(open("bow.sav", 'rb'))
		clean_message=normalizer_all(message)
		a=bow.transform([clean_message])
		svm_bow= pickle.load(open("svm_bow.sav", 'rb'))
		pred=svm_bow.predict_proba(a.toarray())
		my_prediction = pred[0][0]
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)