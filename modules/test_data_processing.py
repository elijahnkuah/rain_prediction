import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from flask import request, Flask,render_template,jsonify
import pickle, os, json
np.random.seed(0)

# Create flask app
IMAGES_FOLDER = os.path.join('./static', 'images')

app = Flask(__name__)
app.config['IMAGES_FOLDER'] = IMAGES_FOLDER
### function to proccess the data
def encode(data, col, max_val):
        data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
        data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
        return data
        
## function to Process the data

def sample_data_processing():
    """ 
    # DATA PREPROCESSING

    **Steps involved in Data Preprocessing:**

    * Label encoding columns with categorical data
    * Perform the scaling of the features
    * Detecting outliers
    * Dropping the outliers based on data analysis"""
    ### Load data

    data = request.get_json()
    if len(data) != 22:
        return({
            "Status":"Failed",
            "Message":"The number {} of input values is not equals to 22".format(len(data))
            })
    else:
        pass
    keys = ['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
       'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm', 'RainToday']
    #input_keys = dict.keys(data)
    for i in dict.keys(data):
        if i not in keys:
            return jsonify({
                "Status":"Failed",
                "Message":f"{i} is not part of the input keys. key names are {keys}"
            })
        else:
            pass
    Date = data["Date"]
    Location = data["Location"]
    MinTemp = data['MinTemp']
    MaxTemp = data["MaxTemp"]
    Rainfall = data["Rainfall"]
    Evaporation = data["Evaporation"]
    Sunshine = data["Sunshine"]
    WindGustDir = data["WindGustDir"]
    WindGustSpeed = data["WindGustSpeed"]
    WindDir9am = data["WindDir9am"]
    WindDir3pm = data["WindDir3pm"]
    WindSpeed9am = data["WindSpeed9am"]
    WindSpeed3pm = data["WindSpeed3pm"]
    Humidity9am = data["Humidity9am"]
    Humidity3pm = data["Humidity3pm"]
    Pressure9am = data["Pressure9am"]
    Pressure3pm =data["Pressure3pm"]
    Cloud9am = data["Cloud9am"]
    Cloud3pm = data["Cloud3pm"]
    Temp9am = data["Temp9am"]
    Temp3pm = data["Temp3pm"]
    RainToday = data["RainToday"]

    # DATA VISUALIZATION AND CLEANING
    data = pd.DataFrame(data, index=[0])

    #There don't seem to be any error in dates so parsing values into datetime
    data['Date']= pd.to_datetime(data["Date"])
    #Creating a collumn of year
    data['year'] = data.Date.dt.year

    # function to encode datetime into cyclic parameters. 
    #As I am planning to use this data in a neural network I prefer the months and days in a cyclic continuous feature. 

    data['month'] = data.Date.dt.month
    data = encode(data, 'month', 12)

    data['day'] = data.Date.dt.day
    data = encode(data, 'day', 31)

    # Get list of categorical variables
    s = (data.dtypes == "object")
    object_cols = list(s[s].index)
    print("Categorical variables:")
    print(object_cols)

    # Get list of neumeric variables
    t = (data.dtypes == "float64")
    num_cols = list(t[t].index)

    print("Numeric variables:")
    print(num_cols)
    # Apply label encoder to each column with categorical data
    # Get list of categorical variables
    s = (data.dtypes == "object")
    object_cols = list(s[s].index)
    print("Categorical variables:")
    print(object_cols)

    ### convert categorical values to numerics
    object_columns = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
    ### Location
    city = ['Adelaide', 'Albany', 'Albury', 'AliceSprings', 'BadgerysCreek',
     'Ballarat', 'Bendigo', 'Brisbane', 'Cairns', 'Canberra', 'Cobar', 
     'CoffsHarbour', 'Dartmoor', 'Darwin', 'GoldCoast', 'Hobart', 'Katherine', 
     'Launceston', 'Melbourne', 'MelbourneAirport', 'Mildura', 'Moree', 'MountGambier', 
     'MountGinini', 'Newcastle', 'Nhil', 'NorahHead', 'NorfolkIsland', 'Nuriootpa', 
     'PearceRAAF', 'Penrith', 'Perth', 'PerthAirport', 'Portland', 'Richmond', 'Sale', 
     'SalmonGums', 'Sydney', 'SydneyAirport', 'Townsville', 'Tuggeranong', 'Uluru', 
     'WaggaWagga', 'Walpole', 'Watsonia', 'Williamtown', 'Witchcliffe', 'Wollongong', 'Woomera']
    number = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 
    45, 46, 47, 48]
    cities_numbers = {}
    cities_numbers["City"]=city
    cities_numbers["Numbers"] = number
    data_cities = pd.DataFrame(cities_numbers)
    location = data["Location"].iloc[0]
    data['Location'] = data_cities.loc[data_cities["City"]==location,"Numbers"].iloc[0]
    ### RainToday
    #today = data["RainToday"].iloc[0]
    if data["RainToday"].iloc[0] == "No":
        data["RainToday"] = 0
    elif data["RainToday"].iloc[0] == "Yes":
        data["RainToday"] = 1
    else:
        return jsonify({
            "Status":"Failled",
            "Message":"The value passed for RainToday {} is incorrrect".format(data["RainToday"])
        })
    ### WindGustDir
    windGustDir = data["WindGustDir"].iloc[0]
    number_wind = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]
    WindGustDir_list = ['E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
        'SSW', 'SW', 'W', 'WNW', 'WSW']
    WindGustDir_data = {}
    WindGustDir_data["Numbers"] = number_wind
    WindGustDir_data["WindGustDir"] = WindGustDir_list
    data_WindGustDir = pd.DataFrame(WindGustDir_data)
    data["WindGustDir"] = data_WindGustDir.loc[data_WindGustDir["WindGustDir"]==windGustDir, "Numbers"].iloc[0]
    
    ###     9am and 3pm
    WindDir3pm_num = [14, 15,  0,  7, 13, 10,  2,  1,  6, 11, 12,  9,  3,  8,  5,  4]
    WindDir9am_num = [13,  6,  9,  1, 12, 10,  8,  4,  3, 11, 15,  2,  0,  7, 14,  5]
    WindDir3pm_cat = ['WNW', 'WSW', 'E', 'NW', 'W', 'SSE', 'ESE', 'ENE', 'NNW', 'SSW',
        'SW', 'SE', 'N', 'S', 'NNE','NE']
    WindDir9am_cat = ['W', 'NNW', 'SE', 'ENE', 'SW', 'SSE', 'S', 'NE','SSW', 'N',
        'WSW', 'ESE', 'E', 'NW', 'WNW', 'NNE']

    data_WindDir3pm = {}
    data_WindDir3pm["Numbers"] = WindDir3pm_num
    data_WindDir3pm["WindDir3pm"] = WindDir3pm_cat
    data_WindDir3pm = pd.DataFrame(data_WindDir3pm)
    data["WindDir3pm"] = data_WindDir3pm.loc[data_WindDir3pm["WindDir3pm"]==data["WindDir3pm"].iloc[0], "Numbers"].iloc[0]


    data_WindDir9am = {}
    data_WindDir9am["Numbers"] = WindDir9am_num
    data_WindDir9am["WindDir9am"] = WindDir9am_cat
    data_WindDir9am = pd.DataFrame(data_WindDir9am)
    data["WindDir9am"] = data_WindDir9am.loc[data_WindDir9am["WindDir9am"]==data["WindDir9am"].iloc[0], "Numbers"].iloc[0]


    # Prepairing attributes of scale data

    features = data.drop(['Date','day', 'month'], axis=1) # dropping target and extra columns
    features = features.astype(float)

    filename_rain = "./models/xgb_model.sav"
    filename_t9am = "./models/xgb_t9_model.sav"
    filename_t3pm = "./models/catr_t3_model.sav"
    filename_c3pm = "./models/adabr_c3_model.sav"
    filename_c9am = "./models/light_c9_model.sav"

    ### Rain Prediction
    cat_model = pickle.load(open(filename_rain, 'rb'))
    rain = cat_model.predict_proba(features)[0][1]
    rain = round(rain,2)
    if rain >=0.8:
        meaning = f"It's stands a higher chance of raining in {Location} Tomorrow"
    elif 0.5 <= rain < 0.8:
        meaning = f"There might be rainfall tomorrow in {Location}"
    elif 0.35 <= rain < 0.5:
        meaning = f"There will be no rainfall at {Location} Tomorrow"
    else:
        meaning = f"There will be serious sunshine tomorrow at {Location}"
    
    data_1 = features
    data_1 = data_1.astype(float)
    Cloud9am_data = data_1.drop(["Cloud9am"],axis=1)
    Cloud3pm_data = data_1.drop(["Cloud3pm"],axis=1)
    Temp3pm_data = data_1.drop(["Temp3pm"],axis=1)
    Temp9am_data = data_1.drop(["Temp9am"],axis=1)
    
    ## Load and predict models 
    xgb_t9 = pickle.load(open(filename_t9am, 'rb'))
    Temp9am = xgb_t9.predict(Temp9am_data)
    Temp9am = round(Temp9am[0],2)
    #Temp9am = str(Temp9am)
    print("Temp at 9am is ", Temp9am)

    cat_t3 = pickle.load(open(filename_t3pm, 'rb'))
    Temp3pm = cat_t3.predict(Temp3pm_data)
    Temp3pm = round(Temp3pm[0],2)
    print("Temp at 3pm is ",Temp3pm)

    ### load and predict Cloud9am model
    Cloud3pm = pickle.load(open(filename_c3pm, 'rb'))
    Cloud3pm = Cloud3pm.predict(Cloud3pm_data)
    Cloud3pm = round(Cloud3pm[0],2)
    print("Cloud 3pm",round(Cloud3pm[0],2))

    Cloud9am = pickle.load(open(filename_c9am, 'rb'))
    Cloud9am = Cloud9am.predict(Cloud9am_data)
    Cloud9am = round(Cloud9am[0],2)
    print("Cloud9am is ",Cloud9am)

    result = {
        "location":Location,
        "rain":rain,
        "message":meaning,
        "Cloud3pm":Cloud3pm,
        "Cloud9pm":Cloud9am,
        "Temp3pm":Temp3pm,
        "Temp9am":Temp9am
        }
    test_data = features.to_json(),
    #return jsonify(location = Location,rain=rain, message=meaning,Cloud3pm=Cloud3pm,Cloud9am=Cloud9am,Temp3pm=Temp3pm,Temp9am=Temp9am)
    return jsonify(result)



    ### WEB APPLICATION
def web_data_processing():
    """ 
    # DATA PREPROCESSING

    **Steps involved in Data Preprocessing:**

    * Label encoding columns with categorical data
    * Perform the scaling of the features
    * Detecting outliers
    * Dropping the outliers based on data analysis"""
    ### Load data

    #data = request.get_json()
    data = json.dumps(request.form)
    data = json.loads(data)
    if len(data) != 22:
        return({
            "Status":"Failed",
            "Message":"The number {} of input values is not equals to 22".format(len(data))
            })
    else:
        pass
    keys = ['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
       'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm', 'RainToday']
    #input_keys = dict.keys(data)
    for i in dict.keys(data):
        if i not in keys:
            return jsonify({
                "Status":"Failed",
                "Message":f"{i} is not part of the input keys. key names are {keys}"
            })
        else:
            pass
    Date = data["Date"]
    Location = data["Location"]
    MinTemp = data['MinTemp']
    MaxTemp = data["MaxTemp"]
    Rainfall = data["Rainfall"]
    Evaporation = data["Evaporation"]
    Sunshine = data["Sunshine"]
    WindGustDir = data["WindGustDir"]
    WindGustSpeed = data["WindGustSpeed"]
    WindDir9am = data["WindDir9am"]
    WindDir3pm = data["WindDir3pm"]
    WindSpeed9am = data["WindSpeed9am"]
    WindSpeed3pm = data["WindSpeed3pm"]
    Humidity9am = data["Humidity9am"]
    Humidity3pm = data["Humidity3pm"]
    Pressure9am = data["Pressure9am"]
    Pressure3pm =data["Pressure3pm"]
    Cloud9am = data["Cloud9am"]
    Cloud3pm = data["Cloud3pm"]
    Temp9am = data["Temp9am"]
    Temp3pm = data["Temp3pm"]
    RainToday = data["RainToday"]

    # DATA VISUALIZATION AND CLEANING
    data = pd.DataFrame(data, index=[0])

    #There don't seem to be any error in dates so parsing values into datetime
    data['Date']= pd.to_datetime(data["Date"])
    #Creating a collumn of year
    data['year'] = data.Date.dt.year

    # function to encode datetime into cyclic parameters. 
    #As I am planning to use this data in a neural network I prefer the months and days in a cyclic continuous feature. 

    data['month'] = data.Date.dt.month
    data = encode(data, 'month', 12)

    data['day'] = data.Date.dt.day
    data = encode(data, 'day', 31)

    # Get list of categorical variables
    s = (data.dtypes == "object")
    object_cols = list(s[s].index)
    print("Categorical variables:")
    print(object_cols)

    # Get list of neumeric variables
    t = (data.dtypes == "float64")
    num_cols = list(t[t].index)

    print("Numeric variables:")
    print(num_cols)
    # Apply label encoder to each column with categorical data
    # Get list of categorical variables
    s = (data.dtypes == "object")
    object_cols = list(s[s].index)
    print("Categorical variables:")
    print(object_cols)

    ### convert categorical values to numerics
    object_columns = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
    ### Location
    city = ['Adelaide', 'Albany', 'Albury', 'AliceSprings', 'BadgerysCreek',
     'Ballarat', 'Bendigo', 'Brisbane', 'Cairns', 'Canberra', 'Cobar', 
     'CoffsHarbour', 'Dartmoor', 'Darwin', 'GoldCoast', 'Hobart', 'Katherine', 
     'Launceston', 'Melbourne', 'MelbourneAirport', 'Mildura', 'Moree', 'MountGambier', 
     'MountGinini', 'Newcastle', 'Nhil', 'NorahHead', 'NorfolkIsland', 'Nuriootpa', 
     'PearceRAAF', 'Penrith', 'Perth', 'PerthAirport', 'Portland', 'Richmond', 'Sale', 
     'SalmonGums', 'Sydney', 'SydneyAirport', 'Townsville', 'Tuggeranong', 'Uluru', 
     'WaggaWagga', 'Walpole', 'Watsonia', 'Williamtown', 'Witchcliffe', 'Wollongong', 'Woomera']
    number = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 
    45, 46, 47, 48]
    cities_numbers = {}
    cities_numbers["City"]=city
    cities_numbers["Numbers"] = number
    data_cities = pd.DataFrame(cities_numbers)
    location = data["Location"].iloc[0]
    data['Location'] = data_cities.loc[data_cities["City"]==location,"Numbers"].iloc[0]
    ### RainToday
    if data["RainToday"].iloc[0] == "" or data["RainToday"].iloc[0] == "No" or data["RainToday"].iloc[0]=="Yes":
        pass
    else:
        return jsonify({
            "Status":"Failled",
            "Message":"The value passed for RainToday is {} is incorrrect".format(data["RainToday"])
        }) 

    if data["RainToday"].iloc[0] == "No":
        data["RainToday"] = 0.0
    elif data["RainToday"].iloc[0] == "Yes":
        data["RainToday"] = 1.0
    else:
        return jsonify({
            "Status":"Failled",
            "Message":"The value passed for RainToday {} is incorrrect".format(data["RainToday"])
        })
        

    
    ### WindGustDir
    windGustDir = data["WindGustDir"].iloc[0]
    number_wind = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]
    WindGustDir_list = ['E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
        'SSW', 'SW', 'W', 'WNW', 'WSW']
    WindGustDir_data = {}
    WindGustDir_data["Numbers"] = number_wind
    WindGustDir_data["WindGustDir"] = WindGustDir_list
    data_WindGustDir = pd.DataFrame(WindGustDir_data)
    data["WindGustDir"] = data_WindGustDir.loc[data_WindGustDir["WindGustDir"]==windGustDir, "Numbers"].iloc[0]
    
    ###     9am and 3pm
    WindDir3pm_num = [14, 15,  0,  7, 13, 10,  2,  1,  6, 11, 12,  9,  3,  8,  5,  4]
    WindDir9am_num = [13,  6,  9,  1, 12, 10,  8,  4,  3, 11, 15,  2,  0,  7, 14,  5]
    WindDir3pm_cat = ['WNW', 'WSW', 'E', 'NW', 'W', 'SSE', 'ESE', 'ENE', 'NNW', 'SSW',
        'SW', 'SE', 'N', 'S', 'NNE','NE']
    WindDir9am_cat = ['W', 'NNW', 'SE', 'ENE', 'SW', 'SSE', 'S', 'NE','SSW', 'N',
        'WSW', 'ESE', 'E', 'NW', 'WNW', 'NNE']

    data_WindDir3pm = {}
    data_WindDir3pm["Numbers"] = WindDir3pm_num
    data_WindDir3pm["WindDir3pm"] = WindDir3pm_cat
    data_WindDir3pm = pd.DataFrame(data_WindDir3pm)
    data["WindDir3pm"] = data_WindDir3pm.loc[data_WindDir3pm["WindDir3pm"]==data["WindDir3pm"].iloc[0], "Numbers"].iloc[0]


    data_WindDir9am = {}
    data_WindDir9am["Numbers"] = WindDir9am_num
    data_WindDir9am["WindDir9am"] = WindDir9am_cat
    data_WindDir9am = pd.DataFrame(data_WindDir9am)
    data["WindDir9am"] = data_WindDir9am.loc[data_WindDir9am["WindDir9am"]==data["WindDir9am"].iloc[0], "Numbers"].iloc[0]


    # Prepairing attributes of scale data

    features = data.drop(['Date','day', 'month'], axis=1) # dropping target and extra columns
    features = features.astype(float)

    filename_rain = "./models/xgb_model.sav"
    filename_t9am = "./models/xgb_t9_model.sav"
    filename_t3pm = "./models/catr_t3_model.sav"
    filename_c3pm = "./models/adabr_c3_model.sav"
    filename_c9am = "./models/light_c9_model.sav"

    ### Rain Prediction
    cat_model = pickle.load(open(filename_rain, 'rb'))
    rain = cat_model.predict_proba(features)[0][1]
    rain = round(rain,2)
    rain = round(rain*100,2)
    if rain >=0.8:
        meaning = "There is {}% Possibility of raining Tomorrow".format(rain)
    elif 0.5 <= rain < 0.8:
        meaning = "There might be rainfall tomorrow with possibility os {}%".format(rain)
    else:
        meaning = "There will be no rainfall Tomorrow since the possibility of raining is {}%".format(rain)

    
    data_1 = features
    data_1 = data_1.astype(float)
    Cloud9am_data = data_1.drop(["Cloud9am"],axis=1)
    Cloud3pm_data = data_1.drop(["Cloud3pm"],axis=1)
    Temp3pm_data = data_1.drop(["Temp3pm"],axis=1)
    Temp9am_data = data_1.drop(["Temp9am"],axis=1)
    
    ## Load and predict models 
    xgb_t9 = pickle.load(open(filename_t9am, 'rb'))
    Temp9am = xgb_t9.predict(Temp9am_data)
    Temp9am = round(Temp9am[0],2)
    #Temp9am = str(Temp9am)
    #print("Temp at 9am is ", Temp9am)

    cat_t3 = pickle.load(open(filename_t3pm, 'rb'))
    Temp3pm = cat_t3.predict(Temp3pm_data)
    Temp3pm = round(Temp3pm[0],2)
    #print("Temp at 3pm is ",Temp3pm)

    ### load and predict Cloud9am model
    Cloud3pm = pickle.load(open(filename_c3pm, 'rb'))
    Cloud3pm = Cloud3pm.predict(Cloud3pm_data)
    Cloud3pm = round(Cloud3pm[0],2)
    #print("Cloud 3pm",round(Cloud3pm[0],2))

    Cloud9am = pickle.load(open(filename_c9am, 'rb'))
    Cloud9am = Cloud9am.predict(Cloud9am_data)
    Cloud9am = round(Cloud9am[0],2)
    #print("Cloud9am is ",Cloud9am)
    full_filename = os.path.join(app.config['IMAGES_FOLDER'], 'whether_aus.PNG')
    plot_data = [
        ("Cloud9am", Cloud9am),
        ("Temp9am", Temp9am),
        ("Cloud3pm", Cloud3pm),
        ("Temp3pm", Temp3pm)
    ]
    labels = [row[0] for row in plot_data]
    values = [row[1] for row in plot_data]

    #test_data = features.to_json()
    #return jsonify(location = Location,rain=rain, message=meaning,Cloud3pm=Cloud3pm,Cloud9am=Cloud9am,Temp3pm=Temp3pm,Temp9am=Temp9am)
    return render_template("report_whether.html",whether_aus=full_filename,location=Location,message=meaning,Cloud3pm=Cloud3pm,
    Cloud9am=Cloud9am,Temp3pm=Temp3pm,Temp9am=Temp9am, labels=labels, values=values)