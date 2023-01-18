import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
np.random.seed(0)

### function to proccess the data
def load_data():
    data_path = "./data/weatherAUS.csv"
    data = pd.read_csv(data_path)
    return data
## function to visualize the data
def visualize():
    data = load_data()
    # DATA VISUALIZATION AND CLEANING

    """**Steps involves in this section:**

    * Count plot of target column 
    * Correlation amongst numeric attributes
    * Parse Dates into datetime
    * Encoding days and months as continuous cyclic features
    """
    #first of all let us evaluate the target and find out if our data is imbalanced or not
    cols= ["#C2C4E2","#EED4E5"]
    sns.countplot(x= data["RainTomorrow"], palette= cols)

    print(data.RainTomorrow.value_counts())
    total_counts = (data.RainTomorrow.value_counts()[0]+data.RainTomorrow.value_counts()[1])
    print("percentage of No is ",(data.RainTomorrow.value_counts()[0]/total_counts)*100)
    print("percentage of Yes is ",(data.RainTomorrow.value_counts()[1]/total_counts)*100)

    # Correlation amongst numeric attributes
    corrmat = data.corr()
    cmap = sns.diverging_palette(260,-10,s=50, l=75, n=6, as_cmap=True)
    plt.subplots(figsize=(18,18))
    sns.heatmap(corrmat,cmap= cmap,annot=True, square=True)

    """**Now I will parse Dates into datetime.**
    My goal is to build an artificial neural network(ANN). 
    I will encode dates appropriately, i.e. I prefer the months 
    and days in a cyclic continuous feature. As, date and time are 
    inherently cyclical. To let the ANN model know that a feature is 
    cyclical I split it into periodic subsections. Namely, years, months
     and days. Now for each subsection, I create two new features, deriving
    a sine transform and cosine transform of the subsection feature. """
    #There don't seem to be any error in dates so parsing values into datetime
    data['Date']= pd.to_datetime(data["Date"])
    #Creating a collumn of year
    data['year'] = data.Date.dt.year

    # function to encode datetime into cyclic parameters. 
    #As I am planning to use this data in a neural network I prefer the months and days in a cyclic continuous feature. 

    def encode(data, col, max_val):
        data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
        data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
        return data

    data['month'] = data.Date.dt.month
    data = encode(data, 'month', 12)

    data['day'] = data.Date.dt.day
    data = encode(data, 'day', 31)

    data.head()

    # roughly a year's span section 
    section = data[:360] 
    tm = section["day"].plot(color="#C2C4E2")
    tm.set_title("Distribution Of Days Over Year")
    tm.set_ylabel("Days In month")
    tm.set_xlabel("Days In Year")

    """As expected, the "year" attribute of data repeats. However in 
    this for the true cyclic nature is not presented in a continuous 
    manner. Splitting months and days into Sine and cosine combination
    provides the cyclical continuous feature. This can be used as input 
    features to ANN. """
    cyclic_month = sns.scatterplot(x="month_sin",y="month_cos",data=data, color="#C2C4E2")
    cyclic_month.set_title("Cyclic Encoding of Month")
    cyclic_month.set_ylabel("Cosine Encoded Months")
    cyclic_month.set_xlabel("Sine Encoded Months")

    plt.figure(figsize=(8,5))
    cyclic_day = sns.scatterplot(x='day_sin',y='day_cos',data=data, color="#C2C4E2")
    cyclic_day.set_title("Cyclic Encoding of Day")
    cyclic_day.set_ylabel("Cosine Encoded Day")
    cyclic_day.set_xlabel("Sine Encoded Day")

    # Get list of categorical variables
    s = (data.dtypes == "object")
    object_cols = list(s[s].index)
    print("Categorical variables:")
    print(object_cols)

    # Filling missing values with mode of the column in value
    for i in object_cols:
        data[i].fillna(data[i].mode()[0], inplace=True)
    
    # Get list of neumeric variables
    t = (data.dtypes == "float64")
    num_cols = list(t[t].index)

    print("Numeric variables:")
    print(num_cols)
    # Missing values in numeric variables
    for i in num_cols:
        print(i, data[i].isnull().sum())
    # Filling missing values with median of the column in value
    for i in num_cols:
        data[i].fillna(data[i].median(), inplace=True)
        
    data.info()

    #plotting a lineplot rainfall over years
    plt.figure(figsize=(12,8))
    Time_series=sns.lineplot(x=data['Date'].dt.year,y="Rainfall",data=data,color="#C2C4E2")
    Time_series.set_title("Rainfall Over Years")
    Time_series.set_ylabel("Rainfall")
    Time_series.set_xlabel("Years")

    #Evauating Wind gust speed over years
    colours = ["#D0DBEE", "#C2C4E2", "#EED4E5", "#D1E6DC", "#BDE2E2"]
    plt.figure(figsize=(12,8))
    Days_of_week=sns.barplot(x=data['Date'].dt.year,y="WindGustSpeed",data=data, ci =None,palette = colours)
    Days_of_week.set_title("Wind Gust Speed Over Years")
    Days_of_week.set_ylabel("WindGustSpeed")
    Days_of_week.set_xlabel("Year")

    return data

def data_processing():
    """ 
    # DATA PREPROCESSING

    **Steps involved in Data Preprocessing:**

    * Label encoding columns with categorical data
    * Perform the scaling of the features
    * Detecting outliers
    * Dropping the outliers based on data analysis"""
    data = visualize()
    # Apply label encoder to each column with categorical data
    # Get list of categorical variables
    s = (data.dtypes == "object")
    object_cols = list(s[s].index)
    print("Categorical variables:")
    print(object_cols)

    label_encoder = LabelEncoder()
    for i in object_cols:
        data[i] = label_encoder.fit_transform(data[i])
        
    data.info()

    # Prepairing attributes of scale data
    ### Independent variables
    features = data.drop(['RainTomorrow', 'Date','day', 'month'], axis=1) # dropping target and extra columns
    features_temp9am = data.drop(['RainTomorrow',"Temp9am", 'Date','day', 'month'], axis=1) # dropping target and extra columns
    features_temp3pm = data.drop(['RainTomorrow',"Temp3pm", 'Date','day', 'month'], axis=1) # dropping target and extra columns
    features_Cloud9am = data.drop(['RainTomorrow',"Cloud9am", 'Date','day', 'month'], axis=1) # dropping target and extra columns
    features_Cloud3pm = data.drop(['RainTomorrow',"Cloud3pm", 'Date','day', 'month'], axis=1) # dropping target and extra columns

    ### Dependent variables 
    target = data['RainTomorrow']
    target_temp9am = data['Temp9am']
    target_temp3pm = data['Temp3pm']
    target_Cloud9am = data['Cloud9am']
    target_Cloud3pm = data['Cloud3pm']

    s_scaler = preprocessing.StandardScaler()

    #Set up a standard scaler for the features
    col_names = list(features.columns)
    features = s_scaler.fit_transform(features)
    features = pd.DataFrame(features, columns=col_names) 

    #Set up a standard scaler for the features_temp9am
    col_names_temp9am = list(features_temp9am.columns)
    features_temp9am = s_scaler.fit_transform(features_temp9am)
    features_temp9am = pd.DataFrame(features_temp9am, columns=col_names_temp9am) 

    #Set up a standard scaler for the features_temp3pm
    col_names_temp3pm = list(features_temp3pm.columns)
    features_temp3pm = s_scaler.fit_transform(features_temp3pm)
    features_temp3pm = pd.DataFrame(features_temp3pm, columns=col_names_temp3pm) 

    #Set up a standard scaler for the features_Cloud9am
    col_names_Cloud9am= list(features_Cloud9am.columns)
    features_Cloud9am = s_scaler.fit_transform(features_Cloud9am)
    features_Cloud9am = pd.DataFrame(features_Cloud9am, columns=col_names_Cloud9am) 

    #Set up a standard scaler for the features_Cloud3pm
    col_names_Cloud3pm = list(features_Cloud3pm.columns)
    features_Cloud3pm = s_scaler.fit_transform(features_Cloud3pm)
    features_Cloud3pm = pd.DataFrame(features_Cloud3pm, columns=col_names_Cloud3pm) 
    #features.describe().T

    #Detecting outliers
    #looking at the scaled features
    colours = ["#D0DBEE", "#C2C4E2", "#EED4E5", "#D1E6DC", "#BDE2E2"]
    plt.figure(figsize=(20,10))
    sns.boxenplot(data = features,palette = colours)
    plt.xticks(rotation=90)
    plt.show()

    #full data for 
    features["RainTomorrow"] = target
    features_temp9am["Temp9am"] = target_temp9am
    features_temp3pm["Temp3pm"] = target_temp3pm
    features_Cloud9am["Cloud9am"] = target_Cloud9am
    features_Cloud3pm["Cloud3Pm"] = target_Cloud3pm                                             

    #Dropping outlier
    features = features[(features["MinTemp"]<2.3)&(features["MinTemp"]>-2.3)]
    features = features[(features["MaxTemp"]<2.3)&(features["MaxTemp"]>-2)]
    features = features[(features["Rainfall"]<4.5)]
    features = features[(features["Evaporation"]<2.8)]
    features = features[(features["Sunshine"]<2.1)]
    features = features[(features["WindGustSpeed"]<4)&(features["WindGustSpeed"]>-4)]
    features = features[(features["WindSpeed9am"]<4)]
    features = features[(features["WindSpeed3pm"]<2.5)]
    features = features[(features["Humidity9am"]>-3)]
    features = features[(features["Humidity3pm"]>-2.2)]
    features = features[(features["Pressure9am"]< 2)&(features["Pressure9am"]>-2.7)]
    features = features[(features["Pressure3pm"]< 2)&(features["Pressure3pm"]>-2.7)]
    features = features[(features["Cloud9am"]<1.8)]
    features = features[(features["Cloud3pm"]<2)]
    features = features[(features["Temp9am"]<2.3)&(features["Temp9am"]>-2)]
    features = features[(features["Temp3pm"]<2.3)&(features["Temp3pm"]>-2)]
    features.shape
    ### Temp9am
    features_temp9am = features_temp9am[(features_temp9am["MinTemp"]<2.3)&(features_temp9am["MinTemp"]>-2.3)]
    features_temp9am = features_temp9am[(features_temp9am["MaxTemp"]<2.3)&(features_temp9am["MaxTemp"]>-2)]
    features_temp9am = features_temp9am[(features_temp9am["Rainfall"]<4.5)]
    features_temp9am = features_temp9am[(features_temp9am["Evaporation"]<2.8)]
    features_temp9am = features_temp9am[(features_temp9am["Sunshine"]<2.1)]
    features_temp9am = features_temp9am[(features_temp9am["WindGustSpeed"]<4)&(features_temp9am["WindGustSpeed"]>-4)]
    features_temp9am = features_temp9am[(features_temp9am["WindSpeed9am"]<4)]
    features_temp9am = features_temp9am[(features_temp9am["WindSpeed3pm"]<2.5)]
    features_temp9am = features_temp9am[(features_temp9am["Humidity9am"]>-3)]
    features_temp9am = features_temp9am[(features_temp9am["Humidity3pm"]>-2.2)]
    features_temp9am = features_temp9am[(features_temp9am["Pressure9am"]< 2)&(features_temp9am["Pressure9am"]>-2.7)]
    features_temp9am = features_temp9am[(features_temp9am["Pressure3pm"]< 2)&(features_temp9am["Pressure3pm"]>-2.7)]
    features_temp9am = features_temp9am[(features_temp9am["Cloud9am"]<1.8)]
    features_temp9am = features_temp9am[(features_temp9am["Cloud3pm"]<2)]
    #features_temp9am = features_temp9am[(features_temp9am["Temp9am"]<2.3)&(features_temp9am["Temp9am"]>-2)]
    features_temp9am = features_temp9am[(features_temp9am["Temp3pm"]<2.3)&(features_temp9am["Temp3pm"]>-2)]

    ### Temp3pm
    features_temp3pm = features_temp3pm[(features_temp3pm["MinTemp"]<2.3)&(features_temp3pm["MinTemp"]>-2.3)]
    features_temp3pm = features_temp3pm[(features_temp3pm["MaxTemp"]<2.3)&(features_temp3pm["MaxTemp"]>-2)]
    features_temp3pm = features_temp3pm[(features_temp3pm["Rainfall"]<4.5)]
    features_temp3pm = features_temp3pm[(features_temp3pm["Evaporation"]<2.8)]
    features_temp3pm = features_temp3pm[(features_temp3pm["Sunshine"]<2.1)]
    features_temp3pm = features_temp3pm[(features_temp3pm["WindGustSpeed"]<4)&(features_temp3pm["WindGustSpeed"]>-4)]
    features_temp3pm = features_temp3pm[(features_temp3pm["WindSpeed9am"]<4)]
    features_temp3pm = features_temp3pm[(features_temp3pm["WindSpeed3pm"]<2.5)]
    features_temp3pm = features_temp3pm[(features_temp3pm["Humidity9am"]>-3)]
    features_temp3pm = features_temp3pm[(features_temp3pm["Humidity3pm"]>-2.2)]
    features_temp3pm = features_temp3pm[(features_temp3pm["Pressure9am"]< 2)&(features_temp3pm["Pressure9am"]>-2.7)]
    features_temp3pm = features_temp3pm[(features_temp3pm["Pressure3pm"]< 2)&(features_temp3pm["Pressure3pm"]>-2.7)]
    features_temp3pm = features_temp3pm[(features_temp3pm["Cloud9am"]<1.8)]
    features_temp3pm = features_temp3pm[(features_temp3pm["Cloud3pm"]<2)]
    features_temp3pm = features_temp3pm[(features_temp3pm["Temp9am"]<2.3)&(features_temp3pm["Temp9am"]>-2)]
    #features_temp3pm = features_temp3pm[(features_temp3pm["Temp3pm"]<2.3)&(features_temp3pm["Temp3pm"]>-2)]

    ### Cloud9am
    features_Cloud9am = features_Cloud9am[(features_Cloud9am["MinTemp"]<2.3)&(features_Cloud9am["MinTemp"]>-2.3)]
    features_Cloud9am = features_Cloud9am[(features_Cloud9am["MaxTemp"]<2.3)&(features_Cloud9am["MaxTemp"]>-2)]
    features_Cloud9am = features_Cloud9am[(features_Cloud9am["Rainfall"]<4.5)]
    features_Cloud9am = features_Cloud9am[(features_Cloud9am["Evaporation"]<2.8)]
    features_Cloud9am = features_Cloud9am[(features_Cloud9am["Sunshine"]<2.1)]
    features_Cloud9am = features_Cloud9am[(features_Cloud9am["WindGustSpeed"]<4)&(features_Cloud9am["WindGustSpeed"]>-4)]
    features_Cloud9am = features_Cloud9am[(features_Cloud9am["WindSpeed9am"]<4)]
    features_Cloud9am = features_Cloud9am[(features_Cloud9am["WindSpeed3pm"]<2.5)]
    features_Cloud9am = features_Cloud9am[(features_Cloud9am["Humidity9am"]>-3)]
    features_Cloud9am = features_Cloud9am[(features_Cloud9am["Humidity3pm"]>-2.2)]
    features_Cloud9am = features_Cloud9am[(features_Cloud9am["Pressure9am"]< 2)&(features_Cloud9am["Pressure9am"]>-2.7)]
    features_Cloud9am = features_Cloud9am[(features_Cloud9am["Pressure3pm"]< 2)&(features_Cloud9am["Pressure3pm"]>-2.7)]
    #features_Cloud9am = features_Cloud9am[(features_Cloud9am["Cloud9am"]<1.8)]
    features_Cloud9am = features_Cloud9am[(features_Cloud9am["Cloud3pm"]<2)]
    features_Cloud9am = features_Cloud9am[(features_Cloud9am["Temp9am"]<2.3)&(features_Cloud9am["Temp9am"]>-2)]
    features_Cloud9am = features_Cloud9am[(features_Cloud9am["Temp3pm"]<2.3)&(features_Cloud9am["Temp3pm"]>-2)]

    ### Cloud3pm
    features_Cloud3pm = features_Cloud3pm[(features_Cloud3pm["MinTemp"]<2.3)&(features_Cloud3pm["MinTemp"]>-2.3)]
    features_Cloud3pm = features_Cloud3pm[(features_Cloud3pm["MaxTemp"]<2.3)&(features_Cloud3pm["MaxTemp"]>-2)]
    features_Cloud3pm = features_Cloud3pm[(features_Cloud3pm["Rainfall"]<4.5)]
    features_Cloud3pm = features_Cloud3pm[(features_Cloud3pm["Evaporation"]<2.8)]
    features_Cloud3pm = features_Cloud3pm[(features_Cloud3pm["Sunshine"]<2.1)]
    features_Cloud3pm = features_Cloud3pm[(features_Cloud3pm["WindGustSpeed"]<4)&(features_Cloud3pm["WindGustSpeed"]>-4)]
    features_Cloud3pm = features_Cloud3pm[(features_Cloud3pm["WindSpeed9am"]<4)]
    features_Cloud3pm = features_Cloud3pm[(features_Cloud3pm["WindSpeed3pm"]<2.5)]
    features_Cloud3pm = features_Cloud3pm[(features_Cloud3pm["Humidity9am"]>-3)]
    features_Cloud3pm = features_Cloud3pm[(features_Cloud3pm["Humidity3pm"]>-2.2)]
    features_Cloud3pm = features_Cloud3pm[(features_Cloud3pm["Pressure9am"]< 2)&(features_Cloud3pm["Pressure9am"]>-2.7)]
    features_Cloud3pm = features_Cloud3pm[(features_Cloud3pm["Pressure3pm"]< 2)&(features_Cloud3pm["Pressure3pm"]>-2.7)]
    features_Cloud3pm = features_Cloud3pm[(features_Cloud3pm["Cloud9am"]<1.8)]
    #features_Cloud3pm = features_Cloud3pm[(features_Cloud3pm["Cloud3pm"]<2)]
    features_Cloud3pm = features_Cloud3pm[(features_Cloud3pm["Temp9am"]<2.3)&(features_Cloud3pm["Temp9am"]>-2)]
    features_Cloud3pm = features_Cloud3pm[(features_Cloud3pm["Temp3pm"]<2.3)&(features_Cloud3pm["Temp3pm"]>-2)]

    #looking at the scaled features without outliers
    plt.figure(figsize=(20,10))
    sns.boxenplot(data = features,palette = colours)
    plt.xticks(rotation=90)
    plt.show()

    # Splitting test and training sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state = 42)

    # Splitting test and training sets
    X_train_t9, X_test_t9, y_train_t9, y_test_t9 = train_test_split(features_temp9am, target_temp9am, test_size = 0.2, random_state = 42)
    # Splitting test and training sets
    X_train_t3, X_test_t3, y_train_t3, y_test_t3 = train_test_split(features_temp3pm, target_temp3pm, test_size = 0.2, random_state = 42)
    # Splitting test and training sets
    X_train_c9, X_test_c9, y_train_c9, y_test_c9 = train_test_split(features_Cloud9am, target_Cloud9am, test_size = 0.2, random_state = 42)
    # Splitting test and training sets
    X_train_c3, X_test_c3, y_train_c3, y_test_c3 = train_test_split(features_Cloud3pm, target_Cloud3pm, test_size = 0.2, random_state = 42)

    #return features