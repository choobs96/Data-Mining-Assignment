from flask import Flask, render_template ,flash, redirect, render_template, request, url_for#this has changed
import plotly
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import os
import tensorflow as tf
import keras
import pandas_datareader.data as web
from datetime import date
from datetime import timedelta,datetime
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
@app.route('/', methods=("POST", "GET"))

def index():
    df2 = create_table()
    line = create_plot()
    line2 = create_plot_2()
    total_line = [line,line2]
    return render_template('index.html', plot=total_line, tables=[df2.to_html(classes='table',formatters={'URL':lambda x:f'<a href="{x}">{x}</a>'}, escape=False)],titles = ['Market Cap'])
def create_plot():


    df= pd.read_csv('Historical_Price.csv',sep="|").drop(['Unnamed: 0'], axis=1)
    #df_1 = df[df['Company Code KL']=="5249.KL"]
    line_name =list(df['Company Code KL'].unique())
    #px.line( x=df["Date"], y=df["Close"], name=df['Company Code KL'])
    #count = 500
    #xScale = np.linspace(0, 100, count)
    #yScale = np.random.randn(count)
    data = []
    for i in line_name:
        df1 = df[df['Company Code KL']==i]
        data.append(go.Scatter(x=df1['Date'], y=df1['Close'],
                    mode='lines',
                    name=i))

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON
    
def create_plot_2():


    klse=pd.read_csv('KLSE_PCT.csv').drop(['Unnamed: 0'], axis=1)
    #df_1 = df[df['Company Code KL']=="5249.KL"]
   
    #px.line( x=df["Date"], y=df["Close"], name=df['Company Code KL'])
    #count = 500
    #xScale = np.linspace(0, 100, count)
    #yScale = np.random.randn(count)
    data_2 = []
    
    data_2.append(go.Scatter(x=klse['Date'], y=klse['Close_P'],
        mode='lines',
        name="KLSE"))

    graphJSON = json.dumps(data_2, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

#@app.route("/")
#def create_table():
#    df2 = pd.read_csv("Property_Market_Cap.csv",sep="|").drop(['Unnamed: 0'], axis=1)
#    df2.set_index(['Company Code KL'], inplace=True)
#    df2.index.name=None
#    return render_template('index.html',tables=[df2.to_html(classes='marketcap')],titles = ['Market Cap'])

def create_table():
    df2 = pd.read_csv("Property_Market_Cap.csv",sep="|").drop(['Unnamed: 0'], axis=1)
    return df2

def create_table_2():
    df3 = pd.read_csv("alphabeta.csv").drop(['Unnamed: 0'], axis=1)
    return df3
   
@app.route('/analysis2')
def analysis2():
    return render_template('analysis2.html')

@app.route('/analysis3')
def analysis3():
	alphabeta  = create_table_2()
	return render_template('analysis3.html',tables=[alphabeta.to_html(classes='table', escape=False)],titles = ['Alpha Beta'])

@app.route('/aboutme')
def aboutme():
    return render_template('aboutme.html')

@app.route('/prediction5200.KL')
def prediction():

    stock = "5200.KL"
    model_5200 = tf.keras.models.load_model('5200_KL.h5')
    apple_quote = pd.read_csv("Prediction_5200.KL.csv")
    apple_quote.set_index(pd.to_datetime(apple_quote['Date']), inplace=True)
    #Create a new dataframe
    new_df = apple_quote.filter(['Close'])
    data = new_df.filter(['Close'])

    #Converting the dataframe to a numpy array
    dataset = data.values

    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaled_data = scaler.fit_transform(dataset)
    #Get teh last 60 day closing price 
    last_n_days = new_df[-60:].values
    #Scale the data to be values between 0 and 1
    last_n_days_scaled = scaler.transform(last_n_days)
    #Create an empty list
    X_test = []
    #Append teh past 60 days
    X_test.append(last_n_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #Get the predicted scaled price
    pred_price = model_5200.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
    pred_price = pred_price[0][0]
    pred_price = round(pred_price,2)
    
    #Get the quote
    apple_quote2 = pd.read_csv("Prediction_today_5200.KL.csv")
    apple_quote2.set_index(pd.to_datetime(apple_quote2['Date']), inplace=True)
    actual = (apple_quote2['Close'].values[0])
    actual = round(actual,2)
    bar,svm_confidence,lr_confidence = create_plot_3()
    valid_5200,rmse = create_table_3()
    return render_template(
        'prediction5200.KL.html',pred_price=pred_price, actual = actual, plot=bar,svm_confidence=svm_confidence,lr_confidence=lr_confidence,tables=[valid_5200.to_html(classes='table', escape=False)],titles = ['5200.KL'],rmse=rmse)


def create_plot_3():
    stock = '5200.KL'
    df = pd.read_csv("Plot_5200.KL.csv")
    df.set_index(pd.to_datetime(df['Date']), inplace=True)
    df=df[['Adj Close']]
    # A variable for predicting 'n' days out into the future
    forecast_out = 30 #'n=30' days
    #Create another column (the target ) shifted 'n' units up
    df['Prediction'] = df[['Adj Close']].shift(-forecast_out)
    ### Create the independent data set (X)  #######
    # Convert the dataframe to a numpy array
    X = np.array(df.drop(['Prediction'],1))

    #Remove the last '30' rows
    X = X[:-forecast_out]
    ### Create the dependent data set (y)  #####
    # Convert the dataframe to a numpy array 
    y = np.array(df['Prediction'])
    # Get all of the y values except the last '30' rows
    y = y[:-forecast_out]
    # Split the data into 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Create and train the Support Vector Machine (Regressor) 
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1) 
    svr_rbf.fit(x_train, y_train)
    # Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
    # The best possible score is 1.0
    svm_confidence = svr_rbf.score(x_test, y_test)

    # Create and train the Linear Regression  Model
    lr = LinearRegression()
    # Train the model
    lr.fit(x_train, y_train)
    # Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
    # The best possible score is 1.0
    lr_confidence = lr.score(x_test, y_test)

    # Set x_forecast equal to the last 30 rows of the original data set from Adj. Close column
    x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
    # Print linear regression model predictions for the next '30' days
    lr_prediction = lr.predict(x_forecast)

    # Print support vector regressor model predictions for the next '30' days
    svm_prediction = svr_rbf.predict(x_forecast)
    df=df.reset_index()
    df=df.append(pd.DataFrame({'Date': pd.date_range(start=df.Date.iloc[-1], periods=31, freq='D', closed='right')}))
    df['SVM_Prediction']=np.nan
    df['LR_Prediction']=np.nan
    df['LR_Prediction'].iloc[-30:]=lr_prediction
    df['SVM_Prediction'].iloc[-30:]=svm_prediction

    trace1=go.Scatter(x=df['Date'], y=df['Adj Close'],
                    mode='lines',
                    name='lines')
    trace2=go.Scatter(x=df['Date'], y=df['LR_Prediction'],
                    mode='lines',
                    name='LR')
    trace3=go.Scatter(x=df['Date'], y=df['SVM_Prediction'],
                    mode='lines', name='SVM')
    data = [trace1,trace2,trace3]
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON,svm_confidence,lr_confidence

@app.route('/prediction1651.KL')
def prediction_2():

    stock = "1651.KL"
    model_5200 = tf.keras.models.load_model('1651_KL.h5')
    apple_quote = pd.read_csv("Prediction_1651.KL.csv")
    apple_quote.set_index(pd.to_datetime(apple_quote['Date']), inplace=True)
    #Create a new dataframe
    new_df = apple_quote.filter(['Close'])
    data = new_df.filter(['Close'])

    #Converting the dataframe to a numpy array
    dataset = data.values

    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaled_data = scaler.fit_transform(dataset)
    #Get teh last 60 day closing price 
    last_n_days = new_df[-60:].values
    #Scale the data to be values between 0 and 1
    last_n_days_scaled = scaler.transform(last_n_days)
    #Create an empty list
    X_test = []
    #Append teh past 60 days
    X_test.append(last_n_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #Get the predicted scaled price
    pred_price = model_5200.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
    pred_price = pred_price[0][0]
    pred_price = round(pred_price,2)
    
    #Get the quote
    apple_quote2 = pd.read_csv("Prediction_today_1651.KL.csv")
    apple_quote2.set_index(pd.to_datetime(apple_quote2['Date']), inplace=True)
    actual = (apple_quote2['Close'].values[0])
    actual = round(actual,2)
    bar,svm_confidence,lr_confidence = create_plot_4()
    valid_5200,rmse = create_table_4()
    return render_template(
        'prediction1651.KL.html',pred_price=pred_price, actual = actual, plot=bar,svm_confidence=svm_confidence,lr_confidence=lr_confidence,tables=[valid_5200.to_html(classes='table', escape=False)],titles = ['1651.KL'],rmse=rmse)

 

def create_plot_4():
    stock = '1651.KL'
    df = pd.read_csv("Plot_1651.KL.csv")
    df.set_index(pd.to_datetime(df['Date']), inplace=True)
    df=df[['Adj Close']]
    # A variable for predicting 'n' days out into the future
    forecast_out = 30 #'n=30' days
    #Create another column (the target ) shifted 'n' units up
    df['Prediction'] = df[['Adj Close']].shift(-forecast_out)
    ### Create the independent data set (X)  #######
    # Convert the dataframe to a numpy array
    X = np.array(df.drop(['Prediction'],1))

    #Remove the last '30' rows
    X = X[:-forecast_out]
    ### Create the dependent data set (y)  #####
    # Convert the dataframe to a numpy array 
    y = np.array(df['Prediction'])
    # Get all of the y values except the last '30' rows
    y = y[:-forecast_out]
    # Split the data into 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Create and train the Support Vector Machine (Regressor) 
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1) 
    svr_rbf.fit(x_train, y_train)
    # Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
    # The best possible score is 1.0
    svm_confidence = svr_rbf.score(x_test, y_test)

    # Create and train the Linear Regression  Model
    lr = LinearRegression()
    # Train the model
    lr.fit(x_train, y_train)
    # Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
    # The best possible score is 1.0
    lr_confidence = lr.score(x_test, y_test)

    # Set x_forecast equal to the last 30 rows of the original data set from Adj. Close column
    x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
    # Print linear regression model predictions for the next '30' days
    lr_prediction = lr.predict(x_forecast)

    # Print support vector regressor model predictions for the next '30' days
    svm_prediction = svr_rbf.predict(x_forecast)
    df=df.reset_index()
    df=df.append(pd.DataFrame({'Date': pd.date_range(start=df.Date.iloc[-1], periods=31, freq='D', closed='right')}))
    df['SVM_Prediction']=np.nan
    df['LR_Prediction']=np.nan
    df['LR_Prediction'].iloc[-30:]=lr_prediction
    df['SVM_Prediction'].iloc[-30:]=svm_prediction

    trace1=go.Scatter(x=df['Date'], y=df['Adj Close'],
                    mode='lines',
                    name='lines')
    trace2=go.Scatter(x=df['Date'], y=df['LR_Prediction'],
                    mode='lines',
                    name='LR')
    trace3=go.Scatter(x=df['Date'], y=df['SVM_Prediction'],
                    mode='lines', name='SVM')
    data = [trace1,trace2,trace3]
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON,svm_confidence,lr_confidence
    
@app.route('/prediction5606.KL')
def prediction_3():

    stock = "5606.KL"
    model_5200 = tf.keras.models.load_model('5606_KL.h5')
    apple_quote = pd.read_csv("Prediction_5606.KL.csv")
    apple_quote.set_index(pd.to_datetime(apple_quote['Date']), inplace=True)
    #Create a new dataframe
    new_df = apple_quote.filter(['Close'])
    data = new_df.filter(['Close'])

    #Converting the dataframe to a numpy array
    dataset = data.values

    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaled_data = scaler.fit_transform(dataset)
    #Get teh last 60 day closing price 
    last_n_days = new_df[-60:].values
    #Scale the data to be values between 0 and 1
    last_n_days_scaled = scaler.transform(last_n_days)
    #Create an empty list
    X_test = []
    #Append teh past 60 days
    X_test.append(last_n_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #Get the predicted scaled price
    pred_price = model_5200.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
    pred_price = pred_price[0][0]
    pred_price = round(pred_price,2)
    
    #Get the quote
    apple_quote2 = pd.read_csv("Prediction_today_5606.KL.csv")
    apple_quote2.set_index(pd.to_datetime(apple_quote2['Date']), inplace=True)
    actual = (apple_quote2['Close'].values[0])
    actual = round(actual,2)
    bar,svm_confidence,lr_confidence = create_plot_5()
    valid_5200,rmse = create_table_5()
    return render_template(
        'prediction5606.KL.html',pred_price=pred_price, actual = actual, plot=bar,svm_confidence=svm_confidence,lr_confidence=lr_confidence,tables=[valid_5200.to_html(classes='table', escape=False)],titles = ['5606.KL'],rmse=rmse)


def create_plot_5():
    stock = '5606.KL'
    df = pd.read_csv("Plot_5606.KL.csv")
    df.set_index(pd.to_datetime(df['Date']), inplace=True)
    df=df[['Adj Close']]
    # A variable for predicting 'n' days out into the future
    forecast_out = 30 #'n=30' days
    #Create another column (the target ) shifted 'n' units up
    df['Prediction'] = df[['Adj Close']].shift(-forecast_out)
    ### Create the independent data set (X)  #######
    # Convert the dataframe to a numpy array
    X = np.array(df.drop(['Prediction'],1))

    #Remove the last '30' rows
    X = X[:-forecast_out]
    ### Create the dependent data set (y)  #####
    # Convert the dataframe to a numpy array 
    y = np.array(df['Prediction'])
    # Get all of the y values except the last '30' rows
    y = y[:-forecast_out]
    # Split the data into 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Create and train the Support Vector Machine (Regressor) 
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1) 
    svr_rbf.fit(x_train, y_train)
    # Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
    # The best possible score is 1.0
    svm_confidence = svr_rbf.score(x_test, y_test)

    # Create and train the Linear Regression  Model
    lr = LinearRegression()
    # Train the model
    lr.fit(x_train, y_train)
    # Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
    # The best possible score is 1.0
    lr_confidence = lr.score(x_test, y_test)

    # Set x_forecast equal to the last 30 rows of the original data set from Adj. Close column
    x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
    # Print linear regression model predictions for the next '30' days
    lr_prediction = lr.predict(x_forecast)

    # Print support vector regressor model predictions for the next '30' days
    svm_prediction = svr_rbf.predict(x_forecast)
    df=df.reset_index()
    df=df.append(pd.DataFrame({'Date': pd.date_range(start=df.Date.iloc[-1], periods=31, freq='D', closed='right')}))
    df['SVM_Prediction']=np.nan
    df['LR_Prediction']=np.nan
    df['LR_Prediction'].iloc[-30:]=lr_prediction
    df['SVM_Prediction'].iloc[-30:]=svm_prediction

    trace1=go.Scatter(x=df['Date'], y=df['Adj Close'],
                    mode='lines',
                    name='lines')
    trace2=go.Scatter(x=df['Date'], y=df['LR_Prediction'],
                    mode='lines',
                    name='LR')
    trace3=go.Scatter(x=df['Date'], y=df['SVM_Prediction'],
                    mode='lines', name='SVM')
    data = [trace1,trace2,trace3]
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON,svm_confidence,lr_confidence
    
@app.route('/prediction3158.KL')
def prediction_4():

    stock = "3158.KL"
    model_5200 = tf.keras.models.load_model('3158_KL.h5')
    apple_quote = pd.read_csv("Prediction_3158.KL.csv")
    apple_quote.set_index(pd.to_datetime(apple_quote['Date']), inplace=True)
    #Create a new dataframe
    new_df = apple_quote.filter(['Close'])
    data = new_df.filter(['Close'])

    #Converting the dataframe to a numpy array
    dataset = data.values

    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaled_data = scaler.fit_transform(dataset)
    #Get teh last 60 day closing price 
    last_n_days = new_df[-60:].values
    #Scale the data to be values between 0 and 1
    last_n_days_scaled = scaler.transform(last_n_days)
    #Create an empty list
    X_test = []
    #Append teh past 60 days
    X_test.append(last_n_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #Get the predicted scaled price
    pred_price = model_5200.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
    pred_price = pred_price[0][0]
    pred_price = round(pred_price,2)
    
    #Get the quote
    apple_quote2 = pd.read_csv("Prediction_today_3158.KL.csv")
    apple_quote2.set_index(pd.to_datetime(apple_quote2['Date']), inplace=True)
    actual = (apple_quote2['Close'].values[0])
    actual = round(actual,2)
    bar,svm_confidence,lr_confidence = create_plot_6()
    valid_5200,rmse = create_table_6()
    return render_template(
        'prediction3158.KL.html',pred_price=pred_price, actual = actual, plot=bar,svm_confidence=svm_confidence,lr_confidence=lr_confidence,tables=[valid_5200.to_html(classes='table', escape=False)],titles = ['3158.KL'],rmse=rmse)


def create_plot_6():
    stock = '3158.KL'
    df = pd.read_csv("Plot_3158.KL.csv")
    df.set_index(pd.to_datetime(df['Date']), inplace=True)
    df=df[['Adj Close']]
    # A variable for predicting 'n' days out into the future
    forecast_out = 30 #'n=30' days
    #Create another column (the target ) shifted 'n' units up
    df['Prediction'] = df[['Adj Close']].shift(-forecast_out)
    ### Create the independent data set (X)  #######
    # Convert the dataframe to a numpy array
    X = np.array(df.drop(['Prediction'],1))

    #Remove the last '30' rows
    X = X[:-forecast_out]
    ### Create the dependent data set (y)  #####
    # Convert the dataframe to a numpy array 
    y = np.array(df['Prediction'])
    # Get all of the y values except the last '30' rows
    y = y[:-forecast_out]
    # Split the data into 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Create and train the Support Vector Machine (Regressor) 
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1) 
    svr_rbf.fit(x_train, y_train)
    # Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
    # The best possible score is 1.0
    svm_confidence = svr_rbf.score(x_test, y_test)

    # Create and train the Linear Regression  Model
    lr = LinearRegression()
    # Train the model
    lr.fit(x_train, y_train)
    # Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
    # The best possible score is 1.0
    lr_confidence = lr.score(x_test, y_test)

    # Set x_forecast equal to the last 30 rows of the original data set from Adj. Close column
    x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
    # Print linear regression model predictions for the next '30' days
    lr_prediction = lr.predict(x_forecast)

    # Print support vector regressor model predictions for the next '30' days
    svm_prediction = svr_rbf.predict(x_forecast)
    df=df.reset_index()
    df=df.append(pd.DataFrame({'Date': pd.date_range(start=df.Date.iloc[-1], periods=31, freq='D', closed='right')}))
    df['SVM_Prediction']=np.nan
    df['LR_Prediction']=np.nan
    df['LR_Prediction'].iloc[-30:]=lr_prediction
    df['SVM_Prediction'].iloc[-30:]=svm_prediction

    trace1=go.Scatter(x=df['Date'], y=df['Adj Close'],
                    mode='lines',
                    name='lines')
    trace2=go.Scatter(x=df['Date'], y=df['LR_Prediction'],
                    mode='lines',
                    name='LR')
    trace3=go.Scatter(x=df['Date'], y=df['SVM_Prediction'],
                    mode='lines', name='SVM')
    data = [trace1,trace2,trace3]
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON,svm_confidence,lr_confidence
    
@app.route('/prediction5053.KL')
def prediction_5():

    stock = "5053.KL"
    model_5200 = tf.keras.models.load_model('5053_KL.h5')
    apple_quote = pd.read_csv("Prediction_5053.KL.csv")
    apple_quote.set_index(pd.to_datetime(apple_quote['Date']), inplace=True)
    #Create a new dataframe
    new_df = apple_quote.filter(['Close'])
    data = new_df.filter(['Close'])

    #Converting the dataframe to a numpy array
    dataset = data.values

    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaled_data = scaler.fit_transform(dataset)
    #Get teh last 60 day closing price 
    last_n_days = new_df[-60:].values
    #Scale the data to be values between 0 and 1
    last_n_days_scaled = scaler.transform(last_n_days)
    #Create an empty list
    X_test = []
    #Append teh past 60 days
    X_test.append(last_n_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #Get the predicted scaled price
    pred_price = model_5200.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
    pred_price = pred_price[0][0]
    pred_price = round(pred_price,2)
    
    #Get the quote
    apple_quote2 = pd.read_csv("Prediction_today_5053.KL.csv")
    apple_quote2.set_index(pd.to_datetime(apple_quote2['Date']), inplace=True)
    actual = (apple_quote2['Close'].values[0])
    actual = round(actual,2)
    bar,svm_confidence,lr_confidence = create_plot_7()
    valid_5200,rmse = create_table_7()
    return render_template(
        'prediction5053.KL.html',pred_price=pred_price, actual = actual, plot=bar,svm_confidence=svm_confidence,lr_confidence=lr_confidence,tables=[valid_5200.to_html(classes='table', escape=False)],titles = ['5053.KL'],rmse=rmse)

def create_plot_7():
    stock = '5053.KL'
    df = pd.read_csv("Plot_5053.KL.csv")
    df.set_index(pd.to_datetime(df['Date']), inplace=True)
    df=df[['Adj Close']]
    # A variable for predicting 'n' days out into the future
    forecast_out = 30 #'n=30' days
    #Create another column (the target ) shifted 'n' units up
    df['Prediction'] = df[['Adj Close']].shift(-forecast_out)
    ### Create the independent data set (X)  #######
    # Convert the dataframe to a numpy array
    X = np.array(df.drop(['Prediction'],1))

    #Remove the last '30' rows
    X = X[:-forecast_out]
    ### Create the dependent data set (y)  #####
    # Convert the dataframe to a numpy array 
    y = np.array(df['Prediction'])
    # Get all of the y values except the last '30' rows
    y = y[:-forecast_out]
    # Split the data into 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Create and train the Support Vector Machine (Regressor) 
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1) 
    svr_rbf.fit(x_train, y_train)
    # Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
    # The best possible score is 1.0
    svm_confidence = svr_rbf.score(x_test, y_test)

    # Create and train the Linear Regression  Model
    lr = LinearRegression()
    # Train the model
    lr.fit(x_train, y_train)
    # Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
    # The best possible score is 1.0
    lr_confidence = lr.score(x_test, y_test)

    # Set x_forecast equal to the last 30 rows of the original data set from Adj. Close column
    x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
    # Print linear regression model predictions for the next '30' days
    lr_prediction = lr.predict(x_forecast)

    # Print support vector regressor model predictions for the next '30' days
    svm_prediction = svr_rbf.predict(x_forecast)
    df=df.reset_index()
    df=df.append(pd.DataFrame({'Date': pd.date_range(start=df.Date.iloc[-1], periods=31, freq='D', closed='right')}))
    df['SVM_Prediction']=np.nan
    df['LR_Prediction']=np.nan
    df['LR_Prediction'].iloc[-30:]=lr_prediction
    df['SVM_Prediction'].iloc[-30:]=svm_prediction

    trace1=go.Scatter(x=df['Date'], y=df['Adj Close'],
                    mode='lines',
                    name='lines')
    trace2=go.Scatter(x=df['Date'], y=df['LR_Prediction'],
                    mode='lines',
                    name='LR')
    trace3=go.Scatter(x=df['Date'], y=df['SVM_Prediction'],
                    mode='lines', name='SVM')
    data = [trace1,trace2,trace3]
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON,svm_confidence,lr_confidence

@app.route('/prediction8206.KL')
def prediction_7():

    stock = "8206.KL"
    model_5200 = tf.keras.models.load_model('8206_KL.h5')
    apple_quote = pd.read_csv("Prediction_8206.KL.csv")
    apple_quote.set_index(pd.to_datetime(apple_quote['Date']), inplace=True)
    #Create a new dataframe
    new_df = apple_quote.filter(['Close'])
    data = new_df.filter(['Close'])

    #Converting the dataframe to a numpy array
    dataset = data.values

    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaled_data = scaler.fit_transform(dataset)
    #Get teh last 60 day closing price 
    last_n_days = new_df[-60:].values
    #Scale the data to be values between 0 and 1
    last_n_days_scaled = scaler.transform(last_n_days)
    #Create an empty list
    X_test = []
    #Append teh past 60 days
    X_test.append(last_n_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #Get the predicted scaled price
    pred_price = model_5200.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
    pred_price = pred_price[0][0]
    pred_price = round(pred_price,2)
    
    #Get the quote
    apple_quote2 = pd.read_csv("Prediction_today_8206.KL.csv")
    apple_quote2.set_index(pd.to_datetime(apple_quote2['Date']), inplace=True)
    actual = (apple_quote2['Close'].values[0])
    actual = round(actual,2)
    bar,svm_confidence,lr_confidence = create_plot_8()
    valid_5200,rmse = create_table_8()
    return render_template(
        'prediction8206.KL.html',pred_price=pred_price, actual = actual, plot=bar,svm_confidence=svm_confidence,lr_confidence=lr_confidence,tables=[valid_5200.to_html(classes='table', escape=False)],titles = ['8206.KL'],rmse=rmse)

def create_plot_8():
    stock = '8206.KL'
    df = pd.read_csv("Plot_8206.KL.csv")
    df.set_index(pd.to_datetime(df['Date']), inplace=True)
    df=df[['Adj Close']]
    # A variable for predicting 'n' days out into the future
    forecast_out = 30 #'n=30' days
    #Create another column (the target ) shifted 'n' units up
    df['Prediction'] = df[['Adj Close']].shift(-forecast_out)
    ### Create the independent data set (X)  #######
    # Convert the dataframe to a numpy array
    X = np.array(df.drop(['Prediction'],1))

    #Remove the last '30' rows
    X = X[:-forecast_out]
    ### Create the dependent data set (y)  #####
    # Convert the dataframe to a numpy array 
    y = np.array(df['Prediction'])
    # Get all of the y values except the last '30' rows
    y = y[:-forecast_out]
    # Split the data into 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Create and train the Support Vector Machine (Regressor) 
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1) 
    svr_rbf.fit(x_train, y_train)
    # Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
    # The best possible score is 1.0
    svm_confidence = svr_rbf.score(x_test, y_test)

    # Create and train the Linear Regression  Model
    lr = LinearRegression()
    # Train the model
    lr.fit(x_train, y_train)
    # Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
    # The best possible score is 1.0
    lr_confidence = lr.score(x_test, y_test)

    # Set x_forecast equal to the last 30 rows of the original data set from Adj. Close column
    x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
    # Print linear regression model predictions for the next '30' days
    lr_prediction = lr.predict(x_forecast)

    # Print support vector regressor model predictions for the next '30' days
    svm_prediction = svr_rbf.predict(x_forecast)
    df=df.reset_index()
    df=df.append(pd.DataFrame({'Date': pd.date_range(start=df.Date.iloc[-1], periods=31, freq='D', closed='right')}))
    df['SVM_Prediction']=np.nan
    df['LR_Prediction']=np.nan
    df['LR_Prediction'].iloc[-30:]=lr_prediction
    df['SVM_Prediction'].iloc[-30:]=svm_prediction

    trace1=go.Scatter(x=df['Date'], y=df['Adj Close'],
                    mode='lines',
                    name='lines')
    trace2=go.Scatter(x=df['Date'], y=df['LR_Prediction'],
                    mode='lines',
                    name='LR')
    trace3=go.Scatter(x=df['Date'], y=df['SVM_Prediction'],
                    mode='lines', name='SVM')
    data = [trace1,trace2,trace3]
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON,svm_confidence,lr_confidence

@app.route('/prediction8664.KL')
def prediction_8():

    stock = "8664.KL"
    model_5200 = tf.keras.models.load_model('8664_KL.h5')
    apple_quote = pd.read_csv("Prediction_8664.KL.csv")
    apple_quote.set_index(pd.to_datetime(apple_quote['Date']), inplace=True)
    #Create a new dataframe
    new_df = apple_quote.filter(['Close'])
    data = new_df.filter(['Close'])

    #Converting the dataframe to a numpy array
    dataset = data.values

    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaled_data = scaler.fit_transform(dataset)
    #Get teh last 60 day closing price 
    last_n_days = new_df[-60:].values
    #Scale the data to be values between 0 and 1
    last_n_days_scaled = scaler.transform(last_n_days)
    #Create an empty list
    X_test = []
    #Append teh past 60 days
    X_test.append(last_n_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #Get the predicted scaled price
    pred_price = model_5200.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
    pred_price = pred_price[0][0]
    pred_price = round(pred_price,2)
    
    #Get the quote
    apple_quote2 = pd.read_csv("Prediction_today_8664.KL.csv")
    apple_quote2.set_index(pd.to_datetime(apple_quote2['Date']), inplace=True)
    actual = (apple_quote2['Close'].values[0])
    actual = round(actual,2)
    bar,svm_confidence,lr_confidence = create_plot_9()
    valid_5200,rmse = create_table_9()
    return render_template(
        'prediction8664.KL.html',pred_price=pred_price, actual = actual, plot=bar,svm_confidence=svm_confidence,lr_confidence=lr_confidence,tables=[valid_5200.to_html(classes='table', escape=False)],titles = ['8664.KL'],rmse=rmse)

def create_plot_9():
    stock = '8664.KL'
    df = pd.read_csv("Plot_8664.KL.csv")
    df.set_index(pd.to_datetime(df['Date']), inplace=True)
    df=df[['Adj Close']]
    # A variable for predicting 'n' days out into the future
    forecast_out = 30 #'n=30' days
    #Create another column (the target ) shifted 'n' units up
    df['Prediction'] = df[['Adj Close']].shift(-forecast_out)
    ### Create the independent data set (X)  #######
    # Convert the dataframe to a numpy array
    X = np.array(df.drop(['Prediction'],1))

    #Remove the last '30' rows
    X = X[:-forecast_out]
    ### Create the dependent data set (y)  #####
    # Convert the dataframe to a numpy array 
    y = np.array(df['Prediction'])
    # Get all of the y values except the last '30' rows
    y = y[:-forecast_out]
    # Split the data into 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Create and train the Support Vector Machine (Regressor) 
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1) 
    svr_rbf.fit(x_train, y_train)
    # Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
    # The best possible score is 1.0
    svm_confidence = svr_rbf.score(x_test, y_test)

    # Create and train the Linear Regression  Model
    lr = LinearRegression()
    # Train the model
    lr.fit(x_train, y_train)
    # Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
    # The best possible score is 1.0
    lr_confidence = lr.score(x_test, y_test)

    # Set x_forecast equal to the last 30 rows of the original data set from Adj. Close column
    x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
    # Print linear regression model predictions for the next '30' days
    lr_prediction = lr.predict(x_forecast)

    # Print support vector regressor model predictions for the next '30' days
    svm_prediction = svr_rbf.predict(x_forecast)
    df=df.reset_index()
    df=df.append(pd.DataFrame({'Date': pd.date_range(start=df.Date.iloc[-1], periods=31, freq='D', closed='right')}))
    df['SVM_Prediction']=np.nan
    df['LR_Prediction']=np.nan
    df['LR_Prediction'].iloc[-30:]=lr_prediction
    df['SVM_Prediction'].iloc[-30:]=svm_prediction

    trace1=go.Scatter(x=df['Date'], y=df['Adj Close'],
                    mode='lines',
                    name='lines')
    trace2=go.Scatter(x=df['Date'], y=df['LR_Prediction'],
                    mode='lines',
                    name='LR')
    trace3=go.Scatter(x=df['Date'], y=df['SVM_Prediction'],
                    mode='lines', name='SVM')
    data = [trace1,trace2,trace3]
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON,svm_confidence,lr_confidence

def calculate_error(i):
    return i**2
    
def create_table_3():
    df3 = pd.read_csv("5200_valid.csv").drop(['Unnamed: 0'], axis=1)
    df3 = df3.head(10)
    rmse = np.sqrt(df3['Difference'].apply(calculate_error).mean())
    return df3,rmse
   
def create_table_4():
    df3 = pd.read_csv("1651_valid.csv").drop(['Unnamed: 0'], axis=1)
    df3 = df3.head(10)
    rmse = np.sqrt(df3['Difference'].apply(calculate_error).mean())
    return df3,rmse
    
def create_table_5():
    df3 = pd.read_csv("5606_valid.csv").drop(['Unnamed: 0'], axis=1)
    df3 = df3.head(10)
    rmse = np.sqrt(df3['Difference'].apply(calculate_error).mean())
    return df3,rmse
    
def create_table_6():
    df3 = pd.read_csv("3158_valid.csv").drop(['Unnamed: 0'], axis=1)
    df3 = df3.head(10)
    rmse = np.sqrt(df3['Difference'].apply(calculate_error).mean())
    return df3,rmse
    
def create_table_7():
    df3 = pd.read_csv("5053_valid.csv").drop(['Unnamed: 0'], axis=1)
    df3 = df3.head(10)
    rmse = np.sqrt(df3['Difference'].apply(calculate_error).mean())
    return df3,rmse
    
def create_table_8():
    df3 = pd.read_csv("8206_valid.csv").drop(['Unnamed: 0'], axis=1)
    df3 = df3.head(10)
    rmse = np.sqrt(df3['Difference'].apply(calculate_error).mean())
    return df3,rmse

def create_table_9():
    df3 = pd.read_csv("8664_valid.csv").drop(['Unnamed: 0'], axis=1)
    df3 = df3.head(10)
    rmse = np.sqrt(df3['Difference'].apply(calculate_error).mean())
    return df3,rmse

    
if __name__ == '__main__':
	app.jinja_env.auto_reload = True
	app.run()