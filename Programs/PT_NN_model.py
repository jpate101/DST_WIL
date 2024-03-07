# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 09:56:39 2020

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

def Dataset_prepare():
    """
    will read files in outputs and inputs folder and extract information from them 
    
    X = is 2 dimentional X[i,:] contains inputs from files in the inputs folder
    Y = is the total drag those inputs create -  1d array  
    """
    """
    dir_in = "old_data\T4v1\inputs/input"
    dir_out = "old_data\T4v1\outputs/out"
    Num_Speed = 20
    Num_files = 55296
    Num_files = Num_files + 1
    """
    """
    dir_in = "old_data\T5v1\inputs/input"
    dir_out = "old_data\T5v1\outputs/out"
    Num_Speed = 20
    Num_files = 59049
    Num_files = Num_files + 1
    
    """
    """
    dir_in = "old_data\T2v1\inputs/input"
    dir_out = "old_data\T2v1\outputs/out"
    Num_Speed = 40
    Num_files = 8640
    Num_files = Num_files + 1
    """
    """
    dir_in = "old_data\T4v1\inputs/input"
    dir_out = "old_data\T4v1\outputs/out"
    Num_Speed = 20
    Num_files = 55296
    Num_files = Num_files + 1
    """
    
    dir_in = "old_data\T1v1\inputs/input"
    dir_out = "old_data\T1v1\outputs/out"
    Num_Speed = 25
    Num_files = 10353
    Num_files = Num_files + 1
    
    X = []
    #read file for inputs
    #for i in range(1, 10369):
    for i in range(1, Num_files):
        input_file = open(dir_in+str(i)+".mlt","r") 
        
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
        
        for speed in range(1,Num_Speed+1):
        #for speed in range(1,16):    
            #X.append([speed,Gravitational_Acceleration,Water_Density,Water_Kin,Eddy_Kin,Air_Density,Air_Kin,Wind_Speed,Wind_Direction,Hull_Displacement_Volume,Hull_Length,Hull_Draft,Hull_MS1_offsets[0],Hull_MS1_offsets[1],Hull_MS1_offsets[2],Hull_MS1_offsets[3],0,0,0,0,0,0])
            X.append([speed,Hull_Length,Hull_Draft, Hull_Displacement_Volume,1,Hull_MS1_offsets[1],Hull_MS1_offsets[2],Hull_MS1_offsets[3],Trim_Speed,Trim_Angle])
            #20 params
            #X.append([speed,20,Hull_MS1_offsets[1],Hull_MS1_offsets[2],Hull_MS1_offsets[3],Hull_MS1_offsets[4],Hull_MS1_offsets[5],Hull_MS1_offsets[6],Hull_MS1_offsets[7],Hull_MS1_offsets[8],Hull_MS1_offsets[9],Hull_MS1_offsets[10],Hull_MS1_offsets[11],Hull_MS1_offsets[12],Hull_MS1_offsets[13],Hull_MS1_offsets[14],Hull_MS1_offsets[15],Hull_MS1_offsets[16],Hull_MS1_offsets[17],Hull_MS1_offsets[18],Hull_MS1_offsets[19],Hull_MS1_offsets[20]])
            
            
        input_file.close() 
     
    Y = [] 
    Y_wave = []
    for i in range(1, Num_files):
    #for i in range(1, 10369):
        #print(i)
        output_file = open(dir_out+str(i)+".mlt","r") 
        
        output_file.readline()#bypass headings 
        for j in range(1,Num_Speed+1):
        #for j in range(1,16):
            output = output_file.readline()
            output = output.split(",")
            #total 
            #print(output)
            #output = [float(output[1]),float(output[2]),float(output[3])]
            #V
            #output = float(output[1])
            #W
            output = float(output[2])
            #if sum(output) > 8000:
            #if output > 1.75:
                #Y_wave.append(i)
                #print(i)
                #print(sum(output))
                #print("________")
            #Y.append(sum(output))
            Y.append((output))
        
        output_file.close() 
        
        
    #shuffle 
    X, Y = shuffle(X, Y, random_state=None)
    #reduce training data 
    
    
    #X, X_test_bin, Y, Y_test_bin = train_test_split(X,Y, test_size = 0.75, random_state = 0)


    print("x length")
    print(len(X))
    print(np.shape(X))
    print("y length")
    print(len(Y))
    print(np.shape(Y))
    
    X = np.array(X)
    Y = np.array(Y)
    
    
    print(len(Y_wave))
    #print(Y_wave)
    #print(Y)
    return X,Y


def Format_Data():
    pass
def get_sec(time_str):
    """Get Seconds from time."""
    time_str = str(time_str)
    
    h = time_str.split(':')
    return float(h[0]) * 3600 + float(h[1]) * 60 + float(h[2])

if __name__=="__main__":
    
    print(device_lib.list_local_devices())
    
    print("\n_______\n")
    
    print("preparing data from input and output files")
    X,Y = Dataset_prepare()
    print("fin preparing data")
    
    
    #print(X)
    print(len(X))
    print(len(X[0]))
    print("________")     
    #print(Y)
    print(len(Y))
    print("end")
    
    #trainning_data_len = len(Y)*.9
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.999,random_state = None)
    
    
    """
    #reduce training data 
    X_train = np.array_split(X_train, 8)
    Y_train = np.array_split(Y_train, 8)
    
    print("train length -----------")
    print(len(Y_train))
    """    
    x_train = np.array(X_train)
    y_train = np.array(Y_train)

    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    
    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
 
    

    #build lstm classifier - V1
    model = Sequential()
    model.add(LSTM(11,return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(LSTM(11, return_sequences=False))
    model.add(Dense(200))
    model.add(Dense(100))
    model.add(Dense(100))
    model.add(Dense(5))
    model.add(Dense(1))
 
    
    
    """
    #build classifier - V3
    model = Sequential()
    model.add(Dense(8,input_shape=(x_train.shape[1],1)))
    model.add(Reshape((1, x_train.shape[1]*8), input_shape=(x_train.shape[1],8)))
    model.add(Dense(80,activation='relu'))
    model.add(Dense(80,activation='relu'))
    model.add(Dense(1))
    """
    
    
    model.summary()    
    #compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    #train model
    #model.fit(x_train,y_train,batch_size = 1,epochs=1)
    n = 10000
    
    start = datetime.datetime.now()
    model.fit(x_train,y_train,batch_size = 1,epochs=1)
    end = datetime.datetime.now()
    #model.fit(TX,TY,batch_size = 1,epochs=1)
    

    
    '''
    load
    from tensorflow import keras
    model = keras.models.load_model('path/to/location')
    
    save
    model = ...  # Get model (Sequential, Functional Model, or Model subclass)
    model.save('path/to/location')
    '''
    
    #save
    #model = ...  # Get model (Sequential, Functional Model, or Model subclass)
    #model.save('Models/model_pickle_RNN')
    
    ##get predicted price values
    predictions = model.predict(x_test) 
    #print(len(predictions))
    #print(len(predictions[0]))
    #print(len(predictions[0][0]))
    #print(len(predictions[0][0][0]))
    
    #here
    #predictions = predictions.mean(axis=2)
    #predictions = predictions.mean(axis=1)
    
    #print(x_train.shape)
    #print(y_train.shape)
    #print(predictions.shape)
    #print(predictions)

    #scaler = MinMaxScaler()
    #y_test = scaler.fit_transform(y_test.reshape(-1,1)) 
    #predictions = scaler.transform(predictions.reshape(-1,1))
    
    size = list(range(0, len(Y_test)))
    
    plt.figure(figsize=(16,8))
    plt.scatter(size,y_test,120)
    plt.scatter(size,predictions,100)
    plt.xlim((0,29))
    #plt.ylim((0,20))
    plt.legend(['Actual','Predictions'],loc='lower right')
    
    plt.title('Recurrent Neural Network model - T2 - trained on 25% of dataset')
    #plt.title('Feedforward Neural Network model - T4 - trained on 50% of dataset')
    plt.xlabel("Samples - x limits 0 to 29")
    plt.ylabel("Scaled Resistance")
    plt.show()
    
  

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
    #plt.plot(predictions)
    plt.title('T1 - wave resistances')
    plt.xlabel("Samples")
    plt.ylabel("wave Resistance (KN)")
    plt.legend(['Train'],loc='lower right')
    plt.show()
  
         
    
    #predictions = predictions[:100000]#9breaks
    #y_test = y_test[:100000]
    
    
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("-----rsme-----")
    print(rsme)
    
    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    
    print(get_sec(end - start))
    
    print("X==============")
    
    
    
