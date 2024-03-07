# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 02:03:34 2021

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
        #Hull_MS1_offsets = [float(Hull_MS1_offsets[0]),float(Hull_MS1_offsets[1]),float(Hull_MS1_offsets[2]),float(Hull_MS1_offsets[3])]
        
        count_offset = 0
        for i in Hull_MS1_offsets:
            Hull_MS1_offsets[count_offset] = float(Hull_MS1_offsets[count_offset])
            count_offset = count_offset + 1
            
            
        Hull_Displacement_Volume = float(Hull_Displacement_Volume)
        Hull_Length = float(Hull_Length)
        Hull_Draft = float(Hull_Draft)
        
        Trim = Trim.split(",")
        Trim_Speed = float(Trim[0])
        Trim_Angle = float(Trim[1])
        
        for speed in range(1,speed_len):
        #for speed in range(1,16):    
            #20 params
            X.append([speed,20,Hull_MS1_offsets[1],Hull_MS1_offsets[2],Hull_MS1_offsets[3],Hull_MS1_offsets[4],Hull_MS1_offsets[5],Hull_MS1_offsets[6],Hull_MS1_offsets[7],Hull_MS1_offsets[8],Hull_MS1_offsets[9],Hull_MS1_offsets[10],Hull_MS1_offsets[11],Hull_MS1_offsets[12],Hull_MS1_offsets[13],Hull_MS1_offsets[14],Hull_MS1_offsets[15],Hull_MS1_offsets[16],Hull_MS1_offsets[17],Hull_MS1_offsets[18],Hull_MS1_offsets[19],Hull_MS1_offsets[20]])
            
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
    
    dir_base_in = "P:\DST_WIL\programs\old_data\T5v1\inputs"
    dir_base_out = "P:\DST_WIL\programs\old_data\T5v1\outputs"
    dir_2less_in = "P:\DST_WIL\programs\old_data\T5v2\inputs"
    dir_2less_out = "P:\DST_WIL\programs\old_data\T5v2\outputs"
    
    TS = .5
    print("prep data")
    X_base,Y_base = Dataset_prepare(dir_base_in,dir_base_out,59049,20)
    print("base")
    X_2less,Y_2less = Dataset_prepare(dir_base_in,dir_base_out,6561,20)
    
    print("Neural network predictions")
    
    x_train, X_test, Y_train, Y_test = train_test_split(X_base,Y_base, test_size = TS,random_state = None)
    
    #build classifier - V3
    #build classifier - V3
    model = Sequential()
    model.add(Dense(8,input_shape=(x_train.shape[1],1)))
    model.add(Reshape((1, x_train.shape[1]*8), input_shape=(x_train.shape[1],8)))
    model.add(Dense(80,activation='relu'))
    model.add(Dense(80,activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model_base = model 
    model_2less = model 

    
    #build lstm classifier - V1
    model1 = Sequential()
    model1.add(LSTM(11,return_sequences=True,input_shape=(22,1)))
    model1.add(LSTM(11, return_sequences=False))
    model1.add(Dense(200))
    model1.add(Dense(100))
    model1.add(Dense(100))
    model1.add(Dense(5))
    model1.add(Dense(1))
    
    model1.compile(optimizer='adam', loss='mean_squared_error')
    
    model_base_rnn = model1 
    model_2less_rnn = model1 

  
    print("Neural network ann predictions")
    #3 params 
    #data prep 
    X_train, X_test, Y_train, Y_test = train_test_split(X_base,Y_base, test_size = TS,random_state = None)
    
    x_train = np.array(X_train)
    y_train = np.array(Y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    
    #train model
    model_base.fit(x_train,y_train,batch_size = 1,epochs=1)
    #report prediction 
    predictions = model_base.predict(x_test) 
    predictions = predictions.mean(axis=2)
    predictions = predictions.mean(axis=1)
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(y_test.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))
    
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("base fnn")
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
    
     #3 params 
    #data prep 
    X_train, X_test, Y_train, Y_test = train_test_split(X_2less,Y_2less, test_size = TS,random_state = None)
    
    x_train = np.array(X_train)
    y_train = np.array(Y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    
    #train model
    model_base.fit(x_train,y_train,batch_size = 1,epochs=1)
    #report prediction 
    predictions = model_base.predict(x_test) 
    predictions = predictions.mean(axis=2)
    predictions = predictions.mean(axis=1)
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(y_test.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))
    
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("2less fnn")
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
    X_train, X_test, Y_train, Y_test = train_test_split(X_2less,Y_2less, test_size = TS,random_state = None)
    
    x_train = np.array(X_train)
    y_train = np.array(Y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    #train model
    model_base_rnn.fit(x_train,y_train,batch_size = 1,epochs=1)
    #report prediction 
    predictions = model_base_rnn.predict(x_test) 
    #predictions = predictions.mean(axis=2)
    #predictions = predictions.mean(axis=1)
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(y_test.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))
    
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("base rnn ")
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
    
    #3 params 
    #data prep 
    X_train, X_test, Y_train, Y_test = train_test_split(X_2less,Y_2less, test_size = TS,random_state = None)
    
    x_train = np.array(X_train)
    y_train = np.array(Y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    #train model
    model_base_rnn.fit(x_train,y_train,batch_size = 1,epochs=1)
    #report prediction 
    predictions = model_2less_rnn.predict(x_test) 
    #predictions = predictions.mean(axis=2)
    #predictions = predictions.mean(axis=1)
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(y_test.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))
    
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("2less rnn ")
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
    X_train, X_test, Y_train, Y_test = train_test_split(X_base,Y_base, test_size = TS,random_state = None)

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
    print("base dt ")
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
    
    #data prep 
    X_train, X_test, Y_train, Y_test = train_test_split(X_2less,Y_2less, test_size = TS,random_state = None)

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
    print("2less dt ")
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
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_base,Y_base,test_size = TS,random_state = None)

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
    print("base nn")
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
    

    
    X_train, X_test, Y_train, Y_test = train_test_split(X_2less,Y_2less, test_size = TS,random_state = None)

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
    print("2less nn")
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
 