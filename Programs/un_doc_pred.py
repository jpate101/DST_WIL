# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 12:43:11 2021

@author: User
"""

import math
import pandas_datareader as web
import numpy as np 
import pandas as pd
#import sklearn as scaler

from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM
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

import pickle
from joblib import dump, load

import time
import datetime

from tensorflow import keras

def Dataset_prepare():
    """
    will read files in outputs and inputs folder and extract information from them 
    
    X = is 2 dimentional X[i,:] contains inputs from files in the inputs folder
    Y = is the total drag those inputs create -  1d array  
    """
    
    
    dir_in = "old_data\T4v4\inputs/input"
    dir_out = "old_data\T4v4\outputs/out"
    Num_Speed = 20
    Num_Speed = Num_Speed + 1
    Num_files = 10125
    Num_files = Num_files + 1
    
    X = []
    #read file for inputs
    for i in range(1, Num_files):
    #for i in range(1, 1001):
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
        Hull_MS1_offsets = [float(Hull_MS1_offsets[0]),float(Hull_MS1_offsets[1]),float(Hull_MS1_offsets[2]),float(Hull_MS1_offsets[3])]
        
        Hull_Displacement_Volume = float(Hull_Displacement_Volume)
        Hull_Length = float(Hull_Length)
        Hull_Draft = float(Hull_Draft)
        
        Trim = Trim.split(",")
        Trim_Speed = float(Trim[0])
        Trim_Angle = float(Trim[1])
        
        for speed in range(1,Num_Speed):
        #for speed in range(1,16):    
            #X.append([speed,Gravitational_Acceleration,Water_Density,Water_Kin,Eddy_Kin,Air_Density,Air_Kin,Wind_Speed,Wind_Direction,Hull_Displacement_Volume,Hull_Length,Hull_Draft,Hull_MS1_offsets[0],Hull_MS1_offsets[1],Hull_MS1_offsets[2],Hull_MS1_offsets[3],0,0,0,0,0,0])
            X.append([speed,Hull_Length,Hull_Draft, Hull_Displacement_Volume,1,Hull_MS1_offsets[1],Hull_MS1_offsets[2],Hull_MS1_offsets[3],Trim_Speed,Trim_Angle])
        input_file.close() 
     
    Y = []  
    Y_wave = []
    #for i in range(1, 1001):
    for i in range(1, Num_files):
        #print(i)
        output_file = open(dir_out+str(i)+".mlt","r") 
        
        output_file.readline()#bypass headings 
        for j in range(1, Num_Speed):
        #for j in range(1,16):
            output = output_file.readline()
            output = output.split(",")
            #total 
            output = [float(output[1]),float(output[2]),float(output[3])]
            #V
            #output = float(output[1])
            #W
            #output = float(output[2])
            if sum(output) > 10000:
                Y_wave.append(i)
                #print(i)
                #print(sum(output))
                #print("________")
            Y.append(sum(output))
            #Y.append((output))
        
        output_file.close() 
     
        
        
    #shuffle 
    X, Y = shuffle(X, Y, random_state=None)
    #reduce training data 
    
    
    #X, X_test_bin, Y, Y_test_bin = train_test_split(X,Y, test_size = 0.5, random_state = 0)


    print("x length")
    print(len(X))
    print(np.shape(X))
    print("y length")
    print(len(Y))
    print(np.shape(Y))
    
    X = np.array(X)
    Y = np.array(Y)
    
    
    
    #print(X)
    print(len(Y_wave))
    return X,Y

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
    
    TS = .5
    print("\n T4v4 \n")
    
    print("preparing data from input and output files")
    dir_in_1 = "old_data\T4v4\inputs"
    dir_out_1 = "old_data\T4v4\outputs"

    X,Y = Dataset_prepare(dir_in_1,dir_out_1,10125,20)
    X_2 = X
    Y_2 = Y
    
    X_3 = X
    Y_3 = Y
    
    X_4 = X
    Y_4 = Y
    print("fin preparing data")
    
    print("DT model ")

    """
    #save dt model 
    with open('Models/model_pickle','wb') as f:
        pickle.dump(DT,f)
    """
    #load
    with open('Models/model_pickle','rb') as f:
        DT = pickle.load(f)
    
    
    y_pred = DT.predict(X)
    
    mse_DT = MSE(Y,y_pred)
    
    rmse_DT = mse_DT**(1/2)
    
    scaler = MinMaxScaler()
    Y = scaler.fit_transform(Y.reshape(-1,1)) 
    y_pred = scaler.transform(y_pred.reshape(-1,1))
    
    rsme = np.sqrt(np.mean(y_pred-Y)**2)
    #print(rmse_DT)
    mse =  np.mean(abs(y_pred-Y))
    print("-----mse-----")
    print(mse)
    print("----rmse----")
    print(rsme)
    

    
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.xlim((100,150))
    #plt.ylim((0,10))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.xlim((1000,1080))
    #plt.ylim((0,10))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
   
    
    print("NN model ")

    """
    #save dt model 
    with open('Models/model_pickle','wb') as f:
        pickle.dump(DT,f)
    """
    #load
    with open('Models/model_pickle_2','rb') as f:
        NN = pickle.load(f)
    
    
    y_pred = NN.predict(X_2)
    
    scaler = MinMaxScaler()
    Y = scaler.fit_transform(Y_2.reshape(-1,1)) 
    y_pred = scaler.transform(y_pred.reshape(-1,1))
    
    rsme = np.sqrt(np.mean(y_pred-Y)**2)
    #print(rmse_DT)
    mse =  np.mean(abs(y_pred-Y))
    print("-----mse-----")
    print(mse)
    print("----rmse----")
    print(rsme)
    

    
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.xlim((100,150))
    #plt.ylim((0,10))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.xlim((1000,1080))
    #plt.ylim((0,10))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    print("FNN model ")
    
    model = keras.models.load_model('Models/model_pickle_FNN')
    
    X_3 = np.reshape(X_3,(X_3.shape[0],X_3.shape[1],1))
    
    predictions = model.predict(X_3) 
    
    predictions = predictions.mean(axis=2)
    predictions = predictions.mean(axis=1)
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(Y_3.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))

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
         
    

    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("-----rsme-----")
    print(rsme)
    
    print("RNN model ")
    
    model = keras.models.load_model('Models/model_pickle_RNN')
    
    X_4 = np.reshape(X_4,(X_4.shape[0],X_3.shape[1],1))
    
    predictions = model.predict(X_4) 
    
    #predictions = predictions.mean(axis=2)
    #predictions = predictions.mean(axis=1)
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(Y_3.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))

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
         
    

    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("-----rsme-----")
    print(rsme)
    
    print("\n T4v4_7 \n")
    
    print("preparing data from input and output files")
    dir_in_1 = "old_data\T4v4_7\inputs"
    dir_out_1 = "old_data\T4v4_7\outputs"

    X,Y = Dataset_prepare(dir_in_1,dir_out_1,10125,20)
    X_2 = X
    Y_2 = Y
    
    X_3 = X
    Y_3 = Y
    
    X_4 = X
    Y_4 = Y
    print("fin preparing data")
    
    print("DT model ")

    """
    #save dt model 
    with open('Models/model_pickle','wb') as f:
        pickle.dump(DT,f)
    """
    #load
    with open('Models/model_pickle','rb') as f:
        DT = pickle.load(f)
    
    
    y_pred = DT.predict(X)
    
    mse_DT = MSE(Y,y_pred)
    
    rmse_DT = mse_DT**(1/2)
    
    scaler = MinMaxScaler()
    Y = scaler.fit_transform(Y.reshape(-1,1)) 
    y_pred = scaler.transform(y_pred.reshape(-1,1))
    
    rsme = np.sqrt(np.mean(y_pred-Y)**2)
    #print(rmse_DT)
    mse =  np.mean(abs(y_pred-Y))
    print("-----mse-----")
    print(mse)
    print("----rmse----")
    print(rsme)
    

    
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.xlim((100,150))
    #plt.ylim((0,10))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.xlim((1000,1080))
    #plt.ylim((0,10))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
   
    
    print("NN model ")

    """
    #save dt model 
    with open('Models/model_pickle','wb') as f:
        pickle.dump(DT,f)
    """
    #load
    with open('Models/model_pickle_2','rb') as f:
        NN = pickle.load(f)
    
    
    y_pred = NN.predict(X_2)
    
    scaler = MinMaxScaler()
    Y = scaler.fit_transform(Y_2.reshape(-1,1)) 
    y_pred = scaler.transform(y_pred.reshape(-1,1))
    
    rsme = np.sqrt(np.mean(y_pred-Y)**2)
    #print(rmse_DT)
    mse =  np.mean(abs(y_pred-Y))
    print("-----mse-----")
    print(mse)
    print("----rmse----")
    print(rsme)
    

    
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.xlim((100,150))
    #plt.ylim((0,10))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.xlim((1000,1080))
    #plt.ylim((0,10))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    print("FNN model ")
    
    model = keras.models.load_model('Models/model_pickle_FNN')
    
    X_3 = np.reshape(X_3,(X_3.shape[0],X_3.shape[1],1))
    
    predictions = model.predict(X_3) 
    
    predictions = predictions.mean(axis=2)
    predictions = predictions.mean(axis=1)
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(Y_3.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))

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
         
    

    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("-----rsme-----")
    print(rsme)
    
    print("RNN model ")
    
    model = keras.models.load_model('Models/model_pickle_RNN')
    
    X_4 = np.reshape(X_4,(X_4.shape[0],X_4.shape[1],1))
    
    predictions = model.predict(X_4) 
    
    #predictions = predictions.mean(axis=2)
    #predictions = predictions.mean(axis=1)
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(Y_3.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))

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
         
    

    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("-----rsme-----")
    print(rsme)
    
    print("\n T4v4_6_2 \n")
    
    print("preparing data from input and output files")
    dir_in_1 = "old_data\T4v4_6_2\inputs"
    dir_out_1 = "old_data\T4v4_6_2\outputs"

    X,Y = Dataset_prepare(dir_in_1,dir_out_1,10125,20)
    X_2 = X
    Y_2 = Y
    
    X_3 = X
    Y_3 = Y
    
    X_4 = X
    Y_4 = Y
    print("fin preparing data")
    
    print("DT model ")

    """
    #save dt model 
    with open('Models/model_pickle','wb') as f:
        pickle.dump(DT,f)
    """
    #load
    with open('Models/model_pickle','rb') as f:
        DT = pickle.load(f)
    
    
    y_pred = DT.predict(X)
    
    mse_DT = MSE(Y,y_pred)
    
    rmse_DT = mse_DT**(1/2)
    
    scaler = MinMaxScaler()
    Y = scaler.fit_transform(Y.reshape(-1,1)) 
    y_pred = scaler.transform(y_pred.reshape(-1,1))
    
    rsme = np.sqrt(np.mean(y_pred-Y)**2)
    #print(rmse_DT)
    mse =  np.mean(abs(y_pred-Y))
    print("-----mse-----")
    print(mse)
    print("----rmse----")
    print(rsme)
    

    
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.xlim((100,150))
    #plt.ylim((0,10))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.xlim((1000,1080))
    #plt.ylim((0,10))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
   
    
    print("NN model ")

    """
    #save dt model 
    with open('Models/model_pickle','wb') as f:
        pickle.dump(DT,f)
    """
    #load
    with open('Models/model_pickle_2','rb') as f:
        NN = pickle.load(f)
    
    
    y_pred = NN.predict(X_2)
    
    scaler = MinMaxScaler()
    Y = scaler.fit_transform(Y_2.reshape(-1,1)) 
    y_pred = scaler.transform(y_pred.reshape(-1,1))
    
    rsme = np.sqrt(np.mean(y_pred-Y)**2)
    #print(rmse_DT)
    mse =  np.mean(abs(y_pred-Y))
    print("-----mse-----")
    print(mse)
    print("----rmse----")
    print(rsme)
    

    
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.xlim((100,150))
    #plt.ylim((0,10))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.xlim((1000,1080))
    #plt.ylim((0,10))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    print("FNN model ")
    
    model = keras.models.load_model('Models/model_pickle_FNN')
    
    X_3 = np.reshape(X_3,(X_3.shape[0],X_3.shape[1],1))
    
    predictions = model.predict(X_3) 
    
    predictions = predictions.mean(axis=2)
    predictions = predictions.mean(axis=1)
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(Y_3.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))

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
         
    

    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("-----rsme-----")
    print(rsme)
    
    print("RNN model ")
    
    model = keras.models.load_model('Models/model_pickle_RNN')
    
    X_4 = np.reshape(X_4,(X_4.shape[0],X_4.shape[1],1))
    
    predictions = model.predict(X_4) 
    
    #predictions = predictions.mean(axis=2)
    #predictions = predictions.mean(axis=1)
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(Y_3.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))

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
         
    

    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("-----rsme-----")
    print(rsme)
    
    print("\n T4v5 \n")
    
    print("preparing data from input and output files")
    dir_in_1 = "old_data\T4v5\inputs"
    dir_out_1 = "old_data\T4v5\outputs"

    X,Y = Dataset_prepare(dir_in_1,dir_out_1,10125,20)
    X_2 = X
    Y_2 = Y
    
    X_3 = X
    Y_3 = Y
    
    X_4 = X
    Y_4 = Y
    print("fin preparing data")
    
    print("DT model ")

    """
    #save dt model 
    with open('Models/model_pickle','wb') as f:
        pickle.dump(DT,f)
    """
    #load
    with open('Models/model_pickle','rb') as f:
        DT = pickle.load(f)
    
    
    y_pred = DT.predict(X)
    
    mse_DT = MSE(Y,y_pred)
    
    rmse_DT = mse_DT**(1/2)
    
    scaler = MinMaxScaler()
    Y = scaler.fit_transform(Y.reshape(-1,1)) 
    y_pred = scaler.transform(y_pred.reshape(-1,1))
    
    rsme = np.sqrt(np.mean(y_pred-Y)**2)
    #print(rmse_DT)
    mse =  np.mean(abs(y_pred-Y))
    print("-----mse-----")
    print(mse)
    print("----rmse----")
    print(rsme)
    

    
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.xlim((100,150))
    #plt.ylim((0,10))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.xlim((1000,1080))
    #plt.ylim((0,10))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
   
    
    print("NN model ")

    """
    #save dt model 
    with open('Models/model_pickle','wb') as f:
        pickle.dump(DT,f)
    """
    #load
    with open('Models/model_pickle_2','rb') as f:
        NN = pickle.load(f)
    
    
    y_pred = NN.predict(X_2)
    
    scaler = MinMaxScaler()
    Y = scaler.fit_transform(Y_2.reshape(-1,1)) 
    y_pred = scaler.transform(y_pred.reshape(-1,1))
    
    rsme = np.sqrt(np.mean(y_pred-Y)**2)
    #print(rmse_DT)
    mse =  np.mean(abs(y_pred-Y))
    print("-----mse-----")
    print(mse)
    print("----rmse----")
    print(rsme)
    

    
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.xlim((100,150))
    #plt.ylim((0,10))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.xlim((1000,1080))
    #plt.ylim((0,10))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y)
    plt.plot(y_pred)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    print("FNN model ")
    
    model = keras.models.load_model('Models/model_pickle_FNN')
    
    X_3 = np.reshape(X_3,(X_3.shape[0],X_3.shape[1],1))
    
    predictions = model.predict(X_3) 
    
    predictions = predictions.mean(axis=2)
    predictions = predictions.mean(axis=1)
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(Y_3.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))

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
         
    

    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("-----rsme-----")
    print(rsme)
    
    print("RNN model ")
    
    model = keras.models.load_model('Models/model_pickle_RNN')
    
    X_4 = np.reshape(X_4,(X_4.shape[0],X_4.shape[1],1))
    
    predictions = model.predict(X_4) 
    
    #predictions = predictions.mean(axis=2)
    #predictions = predictions.mean(axis=1)
    
    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(Y_3.reshape(-1,1)) 
    predictions = scaler.transform(predictions.reshape(-1,1))

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
         
    

    mse =  np.mean(abs(predictions-y_test))
    print("-----mse-----")
    print(mse)
    
    rsme = np.sqrt(np.mean(predictions-y_test)**2)
    print("-----rsme-----")
    print(rsme)