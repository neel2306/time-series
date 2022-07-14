import streamlit as st
import pandas_datareader as pdr
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from keras.models import load_model
from numpy import array


#Normalizing the data.
def scaling_data(dataset):
    scaler = MinMaxScaler(feature_range=(0,1))
    df1 = scaler.fit_transform(np.array(dataset['Close']).reshape(-1,1))
    return df1, scaler

def split_data(data): 

    '''
    A function to split the data into  train and validation sets.
    '''
    train_size = int(len(data) * 0.70)
    test_size = int(len(data) - train_size)
    test_data = data.iloc[train_size:len(data),:1]
    return test_data

def loading_model():
    model = load_model("C:\Codes\Time-Series\lstm_model.h5")
    return model

def last_output(y,model,n_steps=23):
    x_input=np.array(y[len(y)-n_steps:]).reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    lst_output=[]
    i=0
    while(i<30):
        
        if(len(temp_input))>n_steps:
            x_input=np.array(temp_input[1:])
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1
    return lst_output   

def visuals(data,scaler,output):
    df3=data.tolist()
    df3.extend(output)
    df3=scaler.inverse_transform(df3).tolist()
    return df3


if __name__ == "__main__":
    #Title.
    title_alignment = st.markdown("<h1 style='text-align: center; color: green;'>Forecasting</h1>", unsafe_allow_html=True)

    #Getting user input and the corresponding data.
    user_input = st.text_input('Enter stock ticker', 'INFY')
    df = pdr.get_data_yahoo((user_input))

    #Showing the recent 5 days data.
    st.subheader("Past 5 days data")
    st.write(df.tail(5))

    #Showing closing prices.
    st.subheader('Data Visualisation')
    fig = plt.figure(figsize=(12,6))
    plt.plot(df['Close'], color ='red')
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Closing price')
    st.pyplot(fig)

    test_data = split_data(df)
    df1,scaler = scaling_data(df)
    model = loading_model()
    lst_output = last_output(test_data, model)
    df3 = visuals(df1, scaler, lst_output)

    st.subheader('Forecasted Data')
    fig2 = plt.figure(figsize=(12,6))
    plt.plot(df3)
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Forecasted Closing price')
    st.pyplot(fig2)
