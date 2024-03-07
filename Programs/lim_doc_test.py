# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 08:10:59 2021

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



def get_sec(time_str):
    """Get Seconds from time."""
    time_str = str(time_str)
    
    h = time_str.split(':')
    return float(h[0]) * 3600 + float(h[1]) * 60 + float(h[2])

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
    
    dir_v1_in = "P:\DST_WIL\programs\old_data\T4v1\inputs"
    dir_v1_out = "P:\DST_WIL\programs\old_data\T4v1\outputs"
    
    dir_v2_in = "P:\DST_WIL\programs\old_data\T4v2\inputs"
    dir_v2_out = "P:\DST_WIL\programs\old_data\T4v2\outputs"
    
    dir_v3_in = "P:\DST_WIL\programs\old_data\T4v3\inputs"
    dir_v3_out = "P:\DST_WIL\programs\old_data\T4v3\outputs"
 
    
    TS = .5
    print("data set prep")
    X_v1,Y_v1 = Dataset_prepare(dir_v1_in,dir_v1_out,13824*4,20)
    print("1")
    X_v2,Y_v2 = Dataset_prepare(dir_v2_in,dir_v2_out,10125,20)
    print("1")
    X_v3,Y_v3 = Dataset_prepare(dir_v3_in,dir_v3_out,1024,20)
    print("end data set prep")
    
    
    print("decision tree model")
    #data prep 
    X_train, X_test, Y_train, Y_test = train_test_split(X_v1,Y_v1, test_size = TS,random_state = None)

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
    print("v1l")
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
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_v2,Y_v2, test_size = TS,random_state = None)

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
    print("v2")
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
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_v3,Y_v3, test_size = TS,random_state = None)

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
    print("v3")
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


    X_train, X_test, Y_train, Y_test = train_test_split(X_v1,Y_v1, test_size = TS,random_state = None)

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
    print("v1")
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
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_v2,Y_v2, test_size = TS,random_state = None)

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
    print("v2")
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
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_v3,Y_v3, test_size = TS,random_state = None)

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
    print("v3")
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
    

    print("Neural Netwrok models ")

    #build lstm classifier - V1
    model_RNN = Sequential()
    model_RNN.add(LSTM(11,return_sequences=True,input_shape=(x_train.shape[1],1)))
    model_RNN.add(LSTM(11, return_sequences=False))
    model_RNN.add(Dense(200))
    model_RNN.add(Dense(100))
    model_RNN.add(Dense(100))
    model_RNN.add(Dense(5))
    model_RNN.add(Dense(1))
 
    #build classifier - V3
    model_FNN = Sequential()
    model_FNN.add(Dense(8,input_shape=(x_train.shape[1],1)))
    model_FNN.add(Reshape((1, x_train.shape[1]*8), input_shape=(x_train.shape[1],8)))
    model_FNN.add(Dense(80,activation='relu'))
    model_FNN.add(Dense(80,activation='relu'))
    model_FNN.add(Dense(1))
    
    model_RNN.summary()    
    #compile model
    model_RNN.compile(optimizer='adam', loss='mean_squared_error')
    
    model_FNN.summary()    
    #compile model
    model_FNN.compile(optimizer='adam', loss='mean_squared_error')
    
    print("FNN")


    X_train, X_test, Y_train, Y_test = train_test_split(X_v1,Y_v1, test_size = TS,random_state = None)

    x_train = np.array(X_train)
    y_train = np.array(Y_train)

    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    
    model_FNN.fit(x_train,y_train,batch_size = 1,epochs=1)
    
    predictions = model_FNN.predict(x_test) 
    
    predictions = predictions.mean(axis=2)
    predictions = predictions.mean(axis=1)
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(y_test.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))
    
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((100,150))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    
    plt.title('RNN Model - T2 - Trained on 50%')
    plt.xlabel("Samples - x limits 100 to 150 ")
    plt.ylabel("Scaled Resistance")
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1000,1080))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.title('RNN Model - T2 - Trained on 50%')
    plt.xlabel("Samples - x limits 1000 to 1080 ")
    plt.ylabel("Scaled Resistance")
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
             
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("v1")
    print("rsme: "+str(rsme))
    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    print("________________________")
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_v2,Y_v2, test_size = TS,random_state = None)

    x_train = np.array(X_train)
    y_train = np.array(Y_train)

    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    
    model_FNN.fit(x_train,y_train,batch_size = 1,epochs=1)
    
    predictions = model_FNN.predict(x_test) 
    
    predictions = predictions.mean(axis=2)
    predictions = predictions.mean(axis=1)
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(y_test.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))
    
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((100,150))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    
    plt.title('RNN Model - T2 - Trained on 50%')
    plt.xlabel("Samples - x limits 100 to 150 ")
    plt.ylabel("Scaled Resistance")
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1000,1080))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.title('RNN Model - T2 - Trained on 50%')
    plt.xlabel("Samples - x limits 1000 to 1080 ")
    plt.ylabel("Scaled Resistance")
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
             
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("v2")
    print("rsme: "+str(rsme))
    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    print("________________________")

    X_train, X_test, Y_train, Y_test = train_test_split(X_v3,Y_v3, test_size = TS,random_state = None)

    x_train = np.array(X_train)
    y_train = np.array(Y_train)

    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    
    model_FNN.fit(x_train,y_train,batch_size = 1,epochs=1)
    
    predictions = model_FNN.predict(x_test) 
    
    predictions = predictions.mean(axis=2)
    predictions = predictions.mean(axis=1)
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(y_test.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))
    
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((100,150))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    
    plt.title('RNN Model - T2 - Trained on 50%')
    plt.xlabel("Samples - x limits 100 to 150 ")
    plt.ylabel("Scaled Resistance")
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1000,1080))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.title('RNN Model - T2 - Trained on 50%')
    plt.xlabel("Samples - x limits 1000 to 1080 ")
    plt.ylabel("Scaled Resistance")
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
             
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("v3")
    print("rsme: "+str(rsme))
    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    print("________________________")
    
    
    print("RNN")


    X_train, X_test, Y_train, Y_test = train_test_split(X_v1,Y_v1, test_size = TS,random_state = None)

    x_train = np.array(X_train)
    y_train = np.array(Y_train)

    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    
    model_RNN.fit(x_train,y_train,batch_size = 1,epochs=1)
    
    predictions = model_FNN.predict(x_test) 
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(y_test.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))
    
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((100,150))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    
    plt.title('RNN Model - T2 - Trained on 50%')
    plt.xlabel("Samples - x limits 100 to 150 ")
    plt.ylabel("Scaled Resistance")
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1000,1080))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.title('RNN Model - T2 - Trained on 50%')
    plt.xlabel("Samples - x limits 1000 to 1080 ")
    plt.ylabel("Scaled Resistance")
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
             
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("v1")
    print("rsme: "+str(rsme))
    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    print("________________________")
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_v2,Y_v2, test_size = TS,random_state = None)

    x_train = np.array(X_train)
    y_train = np.array(Y_train)

    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    
    model_RNN.fit(x_train,y_train,batch_size = 1,epochs=1)
    
    predictions = model_FNN.predict(x_test) 
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(y_test.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))
    
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((100,150))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    
    plt.title('RNN Model - T2 - Trained on 50%')
    plt.xlabel("Samples - x limits 100 to 150 ")
    plt.ylabel("Scaled Resistance")
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1000,1080))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.title('RNN Model - T2 - Trained on 50%')
    plt.xlabel("Samples - x limits 1000 to 1080 ")
    plt.ylabel("Scaled Resistance")
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
             
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("v2")
    print("rsme: "+str(rsme))
    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    print("________________________")

    X_train, X_test, Y_train, Y_test = train_test_split(X_v3,Y_v3, test_size = TS,random_state = None)

    x_train = np.array(X_train)
    y_train = np.array(Y_train)

    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    
    model_RNN.fit(x_train,y_train,batch_size = 1,epochs=1)
    
    predictions = model_FNN.predict(x_test) 
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(y_test.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))
    
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((100,150))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    
    plt.title('RNN Model - T2 - Trained on 50%')
    plt.xlabel("Samples - x limits 100 to 150 ")
    plt.ylabel("Scaled Resistance")
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test)
    plt.plot(predictions)
    plt.xlim((1000,1080))
    #plt.ylim((0,2))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.title('RNN Model - T2 - Trained on 50%')
    plt.xlabel("Samples - x limits 1000 to 1080 ")
    plt.ylabel("Scaled Resistance")
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
             
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("v3")
    print("rsme: "+str(rsme))
    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    print("________________________")

