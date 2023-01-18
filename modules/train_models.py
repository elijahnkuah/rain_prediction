import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Dense, BatchNormalization, Dropout, LSTM
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import regularizers
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from keras import callbacks
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
import pickle
from train_data_processing import *

"""# MODEL BUILDING


**In this project, we build an artificial neural network.**

**Following steps are involved in the model building**

* Assigning X and y the status of attributes and tags
* Splitting test and training sets
* Initialising the neural network
* Defining by adding layers
* Compiling the neural network
* Train the neural network"""
def split_data():
    features = data_processing()
    X = features.drop(["RainTomorrow"], axis=1)
    y = features["RainTomorrow"]

    # Splitting test and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    X.shape
    return X_train, X_test, y_train, y_test 

### XGBOOST
def xgboost_model():
    xgbc = XGBClassifier(objective='binary:logistic')
    X_train, X_test, y_train, y_test = split_data[0],split_data[1],split_data[2]
    xgbc.fit(X_train,y_train)
    predicted = xgbc.predict(X_test)
    print ("The accuracy of XGBOOST is : ", accuracy_score(y_test, predicted)*100, "%")
    print()
    print("F1 score for XGBoost is :",f1_score(y_test, predicted,)*100, "%")
    print(classification_report(y_test, predicted))
    # confusion matrix
    cmap1 = sns.diverging_palette(260,-10,s=50, l=75, n=5, as_cmap=True)
    plt.subplots(figsize=(12,8))
    cf_matrix = confusion_matrix(y_test, predicted)
    sns.heatmap(cf_matrix/np.sum(cf_matrix), cmap = cmap1, annot = True, annot_kws = {'size':15})
    # save the model to disk
    filename = './models/xgb_model.sav'
    pickle.dump(xgbc, open(filename, 'wb'))
    #load the model from disk
    xgb_model = pickle.load(open(filename, 'rb'))
    result = xgb_model.score(X_test, y_test)
    print(result)

### CATBOOST
def catboost_model():
    catb = CatBoostClassifier()
    X_train, X_test, y_train, y_test = split_data[0],split_data[1],split_data[2]
    catb.fit(X_train,y_train)
    predicted = catb.predict(X_test)
    print ("The accuracy of CATBOOST is : ", accuracy_score(y_test, predicted)*100, "%")
    print()
    print("F1 score for CATBOOST is :",f1_score(y_test, predicted,)*100, "%")
    print(classification_report(y_test, predicted))
    # confusion matrix
    cmap1 = sns.diverging_palette(260,-10,s=50, l=75, n=5, as_cmap=True)
    plt.subplots(figsize=(12,8))
    cf_matrix = confusion_matrix(y_test, predicted)
    sns.heatmap(cf_matrix/np.sum(cf_matrix), cmap = cmap1, annot = True, annot_kws = {'size':15})
    # save the model to disk
    filename = 'models/cat_model.sav'
    pickle.dump(catb, open(filename, 'wb'))
    #load the model from disk
    cat_model = pickle.load(open(filename, 'rb'))
    result = cat_model.score(X_test, y_test)
    print(result)
    feature_importance = pd.DataFrame({'feature_importance': catb.get_feature_importance(), 
              'feature_names': X_train.columns}).sort_values(by=['feature_importance'], 
                                                       ascending=False)

### LIGHTGBM
def lightgbm_model():
    light = LGBMClassifier()
    X_train, X_test, y_train, y_test = split_data[0],split_data[1],split_data[2]
    light.fit(X_train,y_train)
    predicted = light.predict(X_test)
    print ("The accuracy of Lightgbm classifier is : ", accuracy_score(y_test, predicted)*100, "%")
    print()
    print("F1 score for Lightgmb is :",f1_score(y_test, predicted,)*100, "%")
    print(classification_report(y_test, predicted))
    # confusion matrix
    cmap1 = sns.diverging_palette(260,-10,s=50, l=75, n=5, as_cmap=True)
    plt.subplots(figsize=(12,8))
    cf_matrix = confusion_matrix(y_test, predicted)
    sns.heatmap(cf_matrix/np.sum(cf_matrix), cmap = cmap1, annot = True, annot_kws = {'size':15})
    # save the model to disk
    filename = 'models/light_model.sav'
    pickle.dump(light, open(filename, 'wb'))
    #load the model from disk
    light_model = pickle.load(open(filename, 'rb'))
    result = light_model.score(X_test, y_test)
    print(result)

### Deep Learning model
def deep_learning_model():
    #Early stopping
    X_train, X_test, y_train, y_test = split_data[0],split_data[1],split_data[2]
    early_stopping = callbacks.EarlyStopping(
        min_delta=0.001, # minimium amount of change to count as an improvement
        patience=20, # how many epochs to wait before stopping
        restore_best_weights=True,
    )

    # Initialising the NN
    model = Sequential()

    # layers

    model.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu', input_dim = 26))
    model.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    # Compiling the ANN
    opt = Adam(learning_rate=0.00009)
    model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Train the ANN
    history = model.fit(X_train, y_train, batch_size = 32, epochs = 150, callbacks=[early_stopping], validation_split=0.2)

    # Plotting training and validation loss over epochs
    history_df = pd.DataFrame(history.history)
    plt.plot(history_df.loc[:, ['loss']], "#BDE2E2", label='Training loss')
    plt.plot(history_df.loc[:, ['val_loss']],"#C2C4E2", label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc="best")

    plt.show()
    #Plotting training and validation accuracy over epochs
    history_df = pd.DataFrame(history.history)
    plt.plot(history_df.loc[:, ['accuracy']], "#BDE2E2", label='Training accuracy')
    plt.plot(history_df.loc[:, ['val_accuracy']], "#C2C4E2", label='Validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    ### Testing sample data
    """# CONCLUSIONS
    **Concluding the model with:**
    * Testing on the test set
    * Evaluating the confusion matrix
    * Evaluating the classification report"""
    # Predicting the test set results
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)
    # confusion matrix
    cmap1 = sns.diverging_palette(260,-10,s=50, l=75, n=5, as_cmap=True)
    plt.subplots(figsize=(12,8))
    cf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(cf_matrix/np.sum(cf_matrix), cmap = cmap1, annot = True, annot_kws = {'size':15})
    print(classification_report(y_test, y_pred))

def all_models():
    filename_rain = "./models/xgb_model.sav"
    filename_t9am = "./models/xgb_t9_model.sav"
    filename_t3pm = "./models/catr_t3_model.sav"
    filename_c3pm = "./models/adabr_c3_model.sav"
    filename_c9am = "./models/light_c9_model.sav"

    ### Regression model
    xgbr = XGBRegressor()
    model_t9 = xgbr.fit(X_train_t9,y_train_t9)
    print("*************************Temp 9am Done***********************************************")
    catr = CatBoostRegressor()
    model_t3 = catr.fit(X_train_t3,y_train_t3)
    print("*************************Temp 3pm Done**********************************************")
    adab = AdaBoostRegressor()
    model_c3 = adab.fit(X_train_c3,y_train_c3)
    print("*************************Cloud 3pm Done**********************************************")       
    light = LGBMRegressor()
    model_c9 = light.fit(X_train_c9,y_train_c9)
    print("*************************Cloud 9am Done*********************************************")   
            # save the model to disk
    #pickle.dump(rain, open(filename, 'wb'))
    pickle.dump(model_t9, open(filename_t9am,"wb"))
    pickle.dump(model_t3, open(filename_t3pm,"wb"))
    pickle.dump(model_c3, open(filename_c3pm,"wb"))
    pickle.dump(model_c9, open(filename_c9am,"wb"))
