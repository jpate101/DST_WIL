# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 07:14:39 2020

@author: User
"""
import math
import pandas_datareader as web
import numpy as np 
import pandas as pd
#import sklearn as scaler

from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM, Reshape
from keras.models import Sequential

from sklearn.model_selection import train_test_split

from tensorflow.python.client import device_lib

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import time
import datetime

from sklearn.utils import shuffle

from keras.models import Sequential

from sklearn.model_selection import train_test_split

from tensorflow.python.client import device_lib

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from sklearn.utils import shuffle

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import normalize 
from sklearn.preprocessing import MinMaxScaler

def Dataset_prepare(dir_in,dir_out,data_len,speed_len):
    """
    will read files in outputs and inputs folder and extract information from them 
    
    X = is 2 dimentional X[i,:] contains inputs from files in the inputs folder
    Y = is the total drag those inputs create -  1d array  
    """
    X = []
    data_len = data_len + 1
    #read file for inputs
    for i in range(1, data_len):
        input_file = open(dir_in+"/input"+str(i)+".mlt","r") 
        
        for r in range(0, 18):
            input_file.readline()
        Gravitational_Acceleration =  input_file.readline()
        for r in range(0, 2):
            input_file.readline()
        Water_Density = input_file.readline()
        for r in range(0, 1):
            input_file.readline()
        Water_Kin = input_file.readline()
        for r in range(0, 1):
            input_file.readline()
        Eddy_Kin = input_file.readline()
        for r in range(0, 4):
            input_file.readline()
        Air_Density = input_file.readline()  
        for r in range(0, 1):
            input_file.readline()
        Air_Kin = input_file.readline() 
        for r in range(0, 1):
            input_file.readline()
        Wind_Speed = input_file.readline()   
        for r in range(0, 1):
            input_file.readline()
        Wind_Direction = input_file.readline() 
        for r in range(0, 74):
            input_file.readline()
        Hull_MS1_offsets = input_file.readline() 
        for r in range(0, 1):
            input_file.readline()
        Hull_Displacement_Volume = input_file.readline()
        for r in range(0, 1):
            input_file.readline()
        Hull_Length = input_file.readline()
        for r in range(0, 1):
            input_file.readline()
        Hull_Draft = input_file.readline()
        for r in range(0, 14):
            input_file.readline()
        Trim = input_file.readline()
        
        #convert string format to float 
        Gravitational_Acceleration = float(Gravitational_Acceleration)
        Water_Density = float(Water_Density)
        Water_Kin = float(Water_Kin)
        Eddy_Kin = float(Eddy_Kin)
        Air_Density = float(Air_Density)
        Air_Kin = float(Air_Kin)
        Wind_Speed = float(Wind_Speed)
        Wind_Direction = float(Wind_Direction)
        
        Hull_MS1_offsets = Hull_MS1_offsets.split(",")
        Hull_MS1_offsets = [float(Hull_MS1_offsets[0]),float(Hull_MS1_offsets[1]),float(Hull_MS1_offsets[2]),float(Hull_MS1_offsets[3])]
        
        Hull_Displacement_Volume = float(Hull_Displacement_Volume)
        Hull_Length = float(Hull_Length)
        Hull_Draft = float(Hull_Draft)
        
        Trim = Trim.split(",")
        Trim_Speed = float(Trim[0])
        Trim_Angle = float(Trim[1])
        
        for speed in range(1,speed_len):
        #for speed in range(1,16):    
            #X.append([speed,Gravitational_Acceleration,Water_Density,Water_Kin,Eddy_Kin,Air_Density,Air_Kin,Wind_Speed,Wind_Direction,Hull_Displacement_Volume,Hull_Length,Hull_Draft,Hull_MS1_offsets[0],Hull_MS1_offsets[1],Hull_MS1_offsets[2],Hull_MS1_offsets[3],0,0,0,0,0,0])
            X.append([speed,Hull_Length,Hull_Draft, Hull_Displacement_Volume,1,Hull_MS1_offsets[1],Hull_MS1_offsets[2],Hull_MS1_offsets[3],Trim_Speed,Trim_Angle])
        
        input_file.close() 
     
    Y = [] 
    Y_wave = []
    for i in range(1, data_len):
        #print(i)
        output_file = open(dir_out+"/out"+str(i)+".mlt","r") 
        
        output_file.readline()#bypass headings 
        for j in range(1,speed_len):
        #for j in range(1,16):
            output = output_file.readline()
            output = output.split(",")
            #total 
            #print(output)
            output = [float(output[1]),float(output[2]),float(output[3])]
            #V
            #output = float(output[1])
            #W
            #output = float(output[2])
            #if sum(output) > 1:
            #if output > 1.75:
                #Y_wave.append(i)
                #print(i)
                #print(sum(output))
                #print("________")
            Y.append(sum(output))
            #Y.append((output))
        
        output_file.close() 
        
        
    #shuffle 
    X, Y = shuffle(X, Y, random_state=None)
    
    X = np.array(X)
    Y = np.array(Y)

    return X,Y



if __name__=="__main__":
    
    dir_3_in = "P:\DST_WIL\programs\old_data\T3v3\inputs"
    dir_3_out = "P:\DST_WIL\programs\old_data\T3v3\outputs"
    dir_5_in = "P:\DST_WIL\programs\old_data\T3v2\inputs"
    dir_5_out = "P:\DST_WIL\programs\old_data\T3v2\outputs"
    dir_6_in = "P:\DST_WIL\programs\old_data\T3v1\inputs"
    dir_6_out = "P:\DST_WIL\programs\old_data\T3v1\outputs"
    
    dir_7_in = "P:\DST_WIL\programs\old_data\T4v1\inputs"
    dir_7_out = "P:\DST_WIL\programs\old_data\T4v1\outputs"

    TS = .75
    
    #sc_X_3 = StandardScaler()
    #sc_Y_3 = StandardScaler()
    
    #sc_X_5 = StandardScaler()
    #sc_Y_5 = StandardScaler()
    
    #sc_X_6 = StandardScaler()
    #sc_Y_6 = StandardScaler()
    
    #sc_X_7 = StandardScaler()
    #sc_Y_7 = StandardScaler()
    
    print("setting up data transforms")
    
    """
    Xt_3,Yt_3 = Dataset_prepare(dir_3_in,dir_3_out,216,20)
    print("3")
    Xt_5,Yt_5 = Dataset_prepare(dir_5_in,dir_5_out,3456,20)
    print("5")
    Xt_6,Yt_6 = Dataset_prepare(dir_6_in,dir_6_out,13824,20)
    print("6")
    Xt_7,Yt_7 = Dataset_prepare(dir_7_in,dir_7_out,55296,20)
    print("7")
    
    sc_X_3.fit_transform(Xt_3)
    sc_Y_3.fit_transform(Yt_3.reshape(-1,1))
    
    sc_X_5.fit_transform(Xt_5)
    sc_Y_5.fit_transform(Yt_5.reshape(-1,1))
    
    sc_X_6.fit_transform(Xt_6)
    sc_Y_6.fit_transform(Yt_6.reshape(-1,1))
    
    sc_X_7.fit_transform(Xt_7)
    sc_Y_7.fit_transform(Yt_7.reshape(-1,1))
    """
    
    print("Neural network predictions")
    
    
    #build classifier - V3
    #build classifier - V3
    model = Sequential()
    model.add(Dense(8,input_shape=(10,1)))
    model.add(Reshape((1, 10*8), input_shape=(10,8)))
    model.add(Dense(80,activation='relu'))
    model.add(Dense(80,activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model_3 = model 
    model_4 = model 
    model_6 = model 
    model_7 = model 
    
    #build lstm classifier - V1
    model1 = Sequential()
    model1.add(LSTM(11,return_sequences=True,input_shape=(10,1)))
    model1.add(LSTM(11, return_sequences=False))
    model1.add(Dense(200))
    model1.add(Dense(100))
    model1.add(Dense(100))
    model1.add(Dense(5))
    model1.add(Dense(1))
    
    model1.compile(optimizer='adam', loss='mean_squared_error')
    
    model_3_rnn = model1
    model_4_rnn = model1 
    model_6_rnn = model1 
    model_7_rnn = model1 
    
    """
    #build classifiers
    model_3 = Sequential()
    model_3.add(Dense(8,input_shape=(8,1)))
    model_3.add(Reshape((1, 64), input_shape=(8,8)))
    model_3.add(Dense(64,activation='relu'))
    model_3.add(Dense(64,activation='relu'))
    model_3.add(Dense(1))
    model_3.summary()    
    model_3.compile(optimizer='adam', loss='mean_squared_error')
    
    model_4 = Sequential()
    model_4.add(Dense(8,input_shape=(8,1)))
    model_4.add(Reshape((1, 64), input_shape=(8,8)))
    model_4.add(Dense(64,activation='relu'))
    model_4.add(Dense(64,activation='relu'))
    model_4.add(Dense(1))
    model_4.summary()    
    model_4.compile(optimizer='adam', loss='mean_squared_error')
    
    model_6 = Sequential()
    model_6.add(Dense(8,input_shape=(8,1)))
    model_6.add(Reshape((1, 64), input_shape=(8,8)))
    model_6.add(Dense(64,activation='relu'))
    model_6.add(Dense(64,activation='relu'))
    model_6.add(Dense(1))
    model_6.summary()    
    model_6.compile(optimizer='adam', loss='mean_squared_error')
    """
    
  
    print("Neural network ann predictions")
    #3 params 
    #data prep 
    X_3,Y_3 = Dataset_prepare(dir_3_in,dir_3_out,216,20)
    X_train, X_test, Y_train, Y_test = train_test_split(X_3,Y_3, test_size = TS,random_state = None)
    
    x_train = np.array(X_train)
    y_train = np.array(Y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    
    #train model
    model_3.fit(x_train,y_train,batch_size = 1,epochs=1)
    #report prediction 
    predictions = model_3.predict(x_test) 
    predictions = predictions.mean(axis=2)
    predictions = predictions.mean(axis=1)
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(y_test.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))
    
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("3 parms alt in model")
    print("rsme: "+str(rsme))
    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    print("________________________")
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((100,150))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1000,1080))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    #4 params 
    #data prep
    X_4,Y_4 = Dataset_prepare(dir_5_in,dir_5_out,3456,20)
    X_train, X_test, Y_train, Y_test = train_test_split(X_4,Y_4, test_size = TS,random_state = None)
    
    x_train = np.array(X_train)
    y_train = np.array(Y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    #train model
    model_4.fit(x_train,y_train,batch_size = 1,epochs=1)
    #report prediction 
    predictions = model_4.predict(x_test) 
    predictions = predictions.mean(axis=2)
    predictions = predictions.mean(axis=1)
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(y_test.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))
    
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("5 parms alt in model")
    print("rsme: "+str(rsme))
    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    print("________________________")
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((100,150))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1000,1080))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    #6 params 
    #data prep
    X_6,Y_6 = Dataset_prepare(dir_6_in,dir_6_out,13824,20)
    X_train, X_test, Y_train, Y_test = train_test_split(X_6,Y_6, test_size = TS,random_state = None)
    
    x_train = np.array(X_train)
    y_train = np.array(Y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    #train model
    model_6.fit(x_train,y_train,batch_size = 1,epochs=1)
    #report prediction 
    predictions = model_6.predict(x_test) 
    predictions = predictions.mean(axis=2)
    predictions = predictions.mean(axis=1)
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(y_test.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))
    
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("6 parms alt in model")
    print("rsme: "+str(rsme))
    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    print("________________________")
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((100,150))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1000,1080))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    #7 params 
    #data prep
    X_7,Y_7 = Dataset_prepare(dir_7_in,dir_7_out,55296,20)
    X_train, X_test, Y_train, Y_test = train_test_split(X_7,Y_7, test_size = TS,random_state = None)
    
    x_train = np.array(X_train)
    y_train = np.array(Y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    #train model
    model_7.fit(x_train,y_train,batch_size = 1,epochs=1)
    #report prediction 
    predictions = model_7.predict(x_test) 
    predictions = predictions.mean(axis=2)
    predictions = predictions.mean(axis=1)
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(y_test.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))
    
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("7 parms alt in model")
    print("rsme: "+str(rsme))
    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    print("________________________")
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((100,150))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1000,1080))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    print("Neural network rnn predictions")
    #3 params 
    #data prep 
    X_3,Y_3 = Dataset_prepare(dir_3_in,dir_3_out,216,20)
    X_train, X_test, Y_train, Y_test = train_test_split(X_3,Y_3, test_size = TS,random_state = None)
    
    x_train = np.array(X_train)
    y_train = np.array(Y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    #train model
    model_3_rnn.fit(x_train,y_train,batch_size = 1,epochs=1)
    #report prediction 
    predictions = model_3_rnn.predict(x_test) 
    #predictions = predictions.mean(axis=2)
    #predictions = predictions.mean(axis=1)
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(y_test.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))
    
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("3 parms alt in model")
    print("rsme: "+str(rsme))
    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    print("________________________")
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((100,150))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1000,1080))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    #4 params 
    #data prep
    X_4,Y_4 = Dataset_prepare(dir_5_in,dir_5_out,3456,20)
    X_train, X_test, Y_train, Y_test = train_test_split(X_4,Y_4, test_size = TS,random_state = None)

    x_train = np.array(X_train)
    y_train = np.array(Y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    #train model
    model_4.fit(x_train,y_train,batch_size = 1,epochs=1)
    #report prediction 
    predictions = model_4_rnn.predict(x_test) 
    #predictions = predictions.mean(axis=2)
    #predictions = predictions.mean(axis=1)
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(y_test.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))
    
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("5 parms alt in model")
    print("rsme: "+str(rsme))
    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    print("________________________")
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((100,150))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1000,1080))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    #6 params 
    #data prep
    X_6,Y_6 = Dataset_prepare(dir_6_in,dir_6_out,13824,20)
    X_train, X_test, Y_train, Y_test = train_test_split(X_6,Y_6, test_size = TS,random_state = None)

    x_train = np.array(X_train)
    y_train = np.array(Y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    #train model
    model_6.fit(x_train,y_train,batch_size = 1,epochs=1)
    #report prediction 
    predictions = model_6_rnn.predict(x_test) 
    #predictions = predictions.mean(axis=2)
    #predictions = predictions.mean(axis=1)
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(y_test.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))
    
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("6 parms alt in model")
    print("rsme: "+str(rsme))
    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    print("________________________")
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((100,150))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1000,1080))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    #7 params 
    #data prep
    X_7,Y_7 = Dataset_prepare(dir_7_in,dir_7_out,13824*4,20)
    X_train, X_test, Y_train, Y_test = train_test_split(X_6,Y_6, test_size = TS,random_state = None)

    x_train = np.array(X_train)
    y_train = np.array(Y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    #train model
    model_7.fit(x_train,y_train,batch_size = 1,epochs=1)
    #report prediction 
    predictions = model_7_rnn.predict(x_test) 
    #predictions = predictions.mean(axis=2)
    #predictions = predictions.mean(axis=1)
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(y_test.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))
    
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("7 parms alt in model")
    print("rsme: "+str(rsme))
    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    print("________________________")
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((100,150))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1000,1080))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
  
    print("decision tree model")
    #data prep 
    X_3,Y_3 = Dataset_prepare(dir_3_in,dir_3_out,216,20)
    X_train, X_test, Y_train, Y_test = train_test_split(X_3,Y_3, test_size = TS,random_state = None)

    x_train = np.array(X_train)
    y_train = np.array(Y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    #build model
    DT_3 = DecisionTreeRegressor(max_depth=180,min_samples_leaf=1e-10,random_state=None)
    DT_3.fit(X_train,Y_train)
    
    #report prediction 
    predictions = DT_3.predict(X_test) 
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(y_test.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))
    
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("3 parms alt in model")
    print("rsme: "+str(rsme))
    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    print("________________________")
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((100,150))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1000,1080))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    #4 params
    #data prep 
    X_4,Y_4 = Dataset_prepare(dir_5_in,dir_5_out,3456,20)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_4,Y_4, test_size = TS,random_state = None)

    x_train = np.array(X_train)
    y_train = np.array(Y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    #build model
    DT_4 = DecisionTreeRegressor(max_depth=180,min_samples_leaf=1e-10,random_state=None)
    DT_4.fit(X_train,Y_train)
    
    #report prediction 
    predictions = DT_4.predict(X_test) 
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(y_test.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))
    
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("5 parms alt in model")
    print("rsme: "+str(rsme))
    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    print("________________________")
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((100,150))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1000,1080))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    #6 params
    #data prep 
    X_6,Y_6 = Dataset_prepare(dir_6_in,dir_6_out,13824,20)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_6,Y_6, test_size = TS,random_state = None)

    x_train = np.array(X_train)
    y_train = np.array(Y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    #build model
    DT_6 = DecisionTreeRegressor(max_depth=180,min_samples_leaf=1e-10,random_state=None)
    DT_6.fit(X_train,Y_train)
    
    #report prediction 
    predictions = DT_6.predict(X_test) 
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(y_test.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))
    
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("6 parms alt in model")
    print("rsme: "+str(rsme))
    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    print("________________________")
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((100,150))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1000,1080))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    #7 params
    #data prep 
    X_7,Y_7 = Dataset_prepare(dir_7_in,dir_7_out,13824*4,20)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_7,Y_7, test_size = TS,random_state = None)

    x_train = np.array(X_train)
    y_train = np.array(Y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    #build model
    DT_7 = DecisionTreeRegressor(max_depth=180,min_samples_leaf=1e-10,random_state=None)
    DT_7.fit(X_train,Y_train)
    
    #report prediction 
    predictions = DT_7.predict(X_test) 
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(y_test.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))
    
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("7 parms alt in model")
    print("rsme: "+str(rsme))
    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    print("________________________")
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((100,150))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1000,1080))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    
    
    print("Nearest Neighbour model ")

    X_3,Y_3 = Dataset_prepare(dir_3_in,dir_3_out,216,20)
    X_train, X_test, Y_train, Y_test = train_test_split(X_3,Y_3, test_size = TS,random_state = None)

    x_train = np.array(X_train)
    y_train = np.array(Y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    #build model  
    NN = KNeighborsRegressor(n_neighbors=4,weights='distance',algorithm='ball_tree',leaf_size=30)
    NN.fit(X_train,Y_train)
    
    #report prediction 
    predictions  = NN.predict(X_test)
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(y_test.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))
    
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("3 parms alt in model")
    print("rsme: "+str(rsme))
    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    print("________________________")
    
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((100,150))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1000,1080))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    X_4,Y_4 = Dataset_prepare(dir_5_in,dir_5_out,3456,20)
    X_train, X_test, Y_train, Y_test = train_test_split(X_4,Y_4, test_size = TS,random_state = None)

    x_train = np.array(X_train)
    y_train = np.array(Y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    #build model  
    NN = KNeighborsRegressor(n_neighbors=4,weights='distance',algorithm='ball_tree',leaf_size=30)
    NN.fit(X_train,Y_train)
    
    #report prediction 
    predictions  = NN.predict(X_test)
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(y_test.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))
    
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("5 parms alt in model")
    print("rsme: "+str(rsme))
    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    print("________________________")
    
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((100,150))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1000,1080))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    X_6,Y_6 = Dataset_prepare(dir_6_in,dir_6_out,13824,20)
    X_train, X_test, Y_train, Y_test = train_test_split(X_6,Y_6, test_size = TS,random_state = None)

    x_train = np.array(X_train)
    y_train = np.array(Y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    #build model  
    NN = KNeighborsRegressor(n_neighbors=4,weights='distance',algorithm='ball_tree',leaf_size=30)
    NN.fit(X_train,Y_train)
    
    #report prediction 
    predictions  = NN.predict(X_test)
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(y_test.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))
    
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("6 parms alt in model")
    print("rsme: "+str(rsme))
    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    print("________________________")
    
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((100,150))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1000,1080))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    X_7,Y_7 = Dataset_prepare(dir_7_in,dir_7_out,13824*4,20)
    X_train, X_test, Y_train, Y_test = train_test_split(X_7,Y_7, test_size = TS,random_state = None)

    x_train = np.array(X_train)
    y_train = np.array(Y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    #build model  
    NN = KNeighborsRegressor(n_neighbors=4,weights='distance',algorithm='ball_tree',leaf_size=30)
    NN.fit(X_train,Y_train)
    
    #report prediction 
    predictions  = NN.predict(X_test)
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(y_test.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))
    
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("7 parms alt in model")
    print("rsme: "+str(rsme))
    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    print("________________________")
    
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((100,150))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1000,1080))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()

  
    print("Stochastic Gradient Descent ")
    
    X_3,Y_3 = Dataset_prepare(dir_3_in,dir_3_out,216,20)
    X_train, X_test, Y_train, Y_test = train_test_split(X_3,Y_3, test_size = 0.5,random_state = None)
    x_train = np.array(X_train)
    y_train = np.array(Y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    #build model  
    SGD = SGDRegressor(max_iter=100000, tol=1e-3)
    SGD.fit(X_train,Y_train)
    
    #report prediction 
    predictions = SGD.predict(X_test)
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    
    print("3 parms alt in model")
    print("rsme: "+str(rsme))
    print("________________________")
    
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((100,150))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1000,1080))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    X_4,Y_4 = Dataset_prepare(dir_5_in,dir_5_out,3456,20)
    X_train, X_test, Y_train, Y_test = train_test_split(X_4,Y_4, test_size = 0.5,random_state = None)
    x_train = np.array(X_train)
    y_train = np.array(Y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    #build model  
    SGD = SGDRegressor(max_iter=100000, tol=1e-3)
    SGD.fit(X_train,Y_train)
    
    #report prediction 
    predictions = SGD.predict(X_test)
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    
    print("5 parms alt in model")
    print("rsme: "+str(rsme))
    print("________________________")
    
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((100,150))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1000,1080))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    
    X_6,Y_6 = Dataset_prepare(dir_6_in,dir_6_out,13824,20)
    X_train, X_test, Y_train, Y_test = train_test_split(X_6,Y_6, test_size = 0.5,random_state = None)
    x_train = np.array(X_train)
    y_train = np.array(Y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    #build model  
    SGD = SGDRegressor(max_iter=100000, tol=1e-3)
    SGD.fit(X_train,Y_train)
    
    #report prediction 
    predictions = SGD.predict(X_test)
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    
    print("6 parms alt in model")
    print("rsme: "+str(rsme))
    print("________________________")
    
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((100,150))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1000,1080))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    """
    """
    print("SVR model - rbf ")
    
    X_3,Y_3 = Dataset_prepare(dir_3_in,dir_3_out,216,20)
    X_train, X_test, Y_train, Y_test = train_test_split(X_3,Y_3, test_size = 0.5,random_state = None)
    x_train = np.array(X_train)
    y_train = np.array(Y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    #build model  
    SVR = svm.SVR(kernel='rbf', C=2e3,gamma=0.05,cache_size=500 )
    SVR.fit(X_train, Y_train)
    #report prediction 
    predictions = SVR.predict(X_test)
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("3 parms alt in model")
    print("rsme: "+str(rsme))
    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    print("________________________")
    
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((100,150))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1000,1080))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    X_4,Y_4 = Dataset_prepare(dir_5_in,dir_5_out,3456,20)
    X_train, X_test, Y_train, Y_test = train_test_split(X_4,Y_4, test_size = 0.5,random_state = None)
    x_train = np.array(X_train)
    y_train = np.array(Y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    #build model  
    SVR_4 = svm.SVR(kernel='rbf', C=2e3,gamma=0.05,cache_size=500 )
    SVR_4.fit(X_train, Y_train)
    #report prediction 
    predictions = SVR_4.predict(X_test)
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("5 parms alt in model")
    print("rsme: "+str(rsme))
    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    print("________________________")
    
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((100,150))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1000,1080))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    X_6,Y_6 = Dataset_prepare(dir_6_in,dir_6_out,13824,20)
    X_train, X_test, Y_train, Y_test = train_test_split(X_6,Y_6, test_size = 0.5,random_state = None)
    x_train = np.array(X_train)
    y_train = np.array(Y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    #build model  
    SVR_6 = svm.SVR(kernel='rbf', C=2e3,gamma=0.05,cache_size=500 )
    SVR_6.fit(X_train, Y_train)
    #report prediction 
    predictions = SVR_6.predict(X_test)
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("6 parms alt in model")
    print("rsme: "+str(rsme))
    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    print("________________________")
    
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((100,150))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1000,1080))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
 
    pass 
