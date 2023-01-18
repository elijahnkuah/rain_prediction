from flask import Flask, request, render_template,jsonify
import json
import pickle
from modules.test_data_processing import *
app = Flask(__name__)
#app.config[]
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/report_whether.html",methods=["POST"])
def predict_whether():
    prediction_web = web_data_processing()
    ### import saved model
    return prediction_web

@app.route("/result",methods=["POST"])
def predict_rain():
    prediction = sample_data_processing()
    ### import saved model

    return prediction

if __name__ == "__main__":
    app.run(debug=True)