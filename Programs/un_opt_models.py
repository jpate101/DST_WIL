# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 15:21:39 2021

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
from sklearn.preprocessing import PolynomialFeatures

import pickle
from joblib import dump, load

import time
import datetime

from sklearn.datasets import load_iris
from sklearn import tree

def get_sec(time_str):
    """Get Seconds from time."""
    time_str = str(time_str)
    
    h = time_str.split(':')
    return float(h[0]) * 3600 + float(h[1]) * 60 + float(h[2])

def Dataset_prepare():
    """
    will read files in outputs and inputs folder and extract information from them 
    
    X = is 2 dimentional X[i,:] contains inputs from files in the inputs folder
    Y = is the total drag those inputs create -  1d array  
    """
    
    """
    dir_in = "old_data\T2v3\inputs/input"
    dir_out = "old_data\T2v3\outputs/out"
    Num_Speed = 40
    Num_Speed = Num_Speed + 1
    Num_files = 17280
    Num_files = Num_files + 1
    """
    dir_in = "old_data\T4v1\inputs/input"
    dir_out = "old_data\T4v1\outputs/out"
    Num_Speed = 20
    Num_Speed = Num_Speed + 1
    Num_files = 55296
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
        
        for speed in range(1,Num_Speed):
        #for speed in range(1,16):    
            #X.append([speed,Gravitational_Acceleration,Water_Density,Water_Kin,Eddy_Kin,Air_Density,Air_Kin,Wind_Speed,Wind_Direction,Hull_Displacement_Volume,Hull_Length,Hull_Draft,Hull_MS1_offsets[0],Hull_MS1_offsets[1],Hull_MS1_offsets[2],Hull_MS1_offsets[3],0,0,0,0,0,0])
            X.append([speed,Hull_Length,Hull_Draft, Hull_Displacement_Volume,1,Hull_MS1_offsets[1],Hull_MS1_offsets[2],Hull_MS1_offsets[3],Trim_Speed,Trim_Angle])
            #20 params
            #X.append([speed,20,Hull_MS1_offsets[1],Hull_MS1_offsets[2],Hull_MS1_offsets[3],Hull_MS1_offsets[4],Hull_MS1_offsets[5],Hull_MS1_offsets[6],Hull_MS1_offsets[7],Hull_MS1_offsets[8],Hull_MS1_offsets[9],Hull_MS1_offsets[10],Hull_MS1_offsets[11],Hull_MS1_offsets[12],Hull_MS1_offsets[13],Hull_MS1_offsets[14],Hull_MS1_offsets[15],Hull_MS1_offsets[16],Hull_MS1_offsets[17],Hull_MS1_offsets[18],Hull_MS1_offsets[19],Hull_MS1_offsets[20]])
            
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
    
    
    #print(Hull_MS1_offsets)
    
    return X,Y


if __name__=="__main__":
    
    TS = .5
    
    print(device_lib.list_local_devices())
    
    print("\n_______\n")
    
    print("preparing data from input and output files")
    X,Y = Dataset_prepare()
    print("fin preparing data")
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = TS, random_state = 0)
    
  
    print("DT model ")
    start = datetime.datetime.now()
    #DT = DecisionTreeRegressor(max_depth=180,min_samples_leaf=1e-10,criterion='mae',random_state=None)
    DT = DecisionTreeRegressor()
    
    DT.fit(X_train,Y_train)
    end = datetime.datetime.now()
    
    """
    #save dt model 
    with open('Models/model_pickle','wb') as f:
        pickle.dump(DT,f)
    
    #load
    with open('Models/model_pickle','rb') as f:
        DT = pickle.load(f)
    """
    
    y_pred = DT.predict(X_test)
    
    mse_DT = MSE(Y_test,y_pred)
    
    rmse_DT = mse_DT**(1/2)
    
    scaler = MinMaxScaler()
    Y_test = scaler.fit_transform(Y_test.reshape(-1,1)) 
    y_pred = scaler.transform(y_pred.reshape(-1,1))
    
    rsme = np.sqrt(np.mean(y_pred-Y_test)**2)
    #print(rmse_DT)
    mse =  np.mean(abs(y_pred-Y_test))
    print("-----mse-----")
    print(mse)
    print("----rmse----")
    print(rsme)
    
    print(get_sec(end - start))
    
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(Y_test)
    plt.plot(y_pred)
    plt.xlim((100,150))
    #plt.ylim((0,10))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y_test)
    plt.plot(y_pred)
    plt.xlim((1000,1080))
    #plt.ylim((0,10))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y_test)
    plt.plot(y_pred)
    plt.xlim((1200,1380))
    #plt.ylim((0,20))
    plt.legend(['Train','Predictions'],loc='lower right')
    
    plt.title('Decision Tree mode - T3 - trained on 75% of dataset')
    plt.xlabel("Samples - x limits 1200 to 1380 ")
    plt.ylabel("Scaled Resistance")
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y_test)
    plt.plot(y_pred)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.title('plot 1')
    plt.xlabel("X test")
    plt.ylabel("X test")
    plt.show()
  
  
    print("NN model ")
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = TS, random_state = 0)
    
    start = datetime.datetime.now()
    NN = KNeighborsRegressor()
    
    NN.fit(X_train,Y_train)
    end = datetime.datetime.now()
    
    """
    #save dt model 
    with open('Models/model_pickle_2','wb') as f:
        pickle.dump(DT,f)
    
    #load
    with open('Models/model_pickle_2','rb') as f:
        NN = pickle.load(f)
    """
    
    
    y_pred = NN.predict(X_test)
    
    scaler = MinMaxScaler()
    Y_test = scaler.fit_transform(Y_test.reshape(-1,1)) 
    y_pred = scaler.transform(y_pred.reshape(-1,1))
    
    
    
    rsme = np.sqrt(np.mean(y_pred-Y_test)**2)
    mse =  np.mean(abs(y_pred-Y_test))
    print("-----mse-----")
    print(mse)
    print("----rmse----")
    print(rsme)

    print(get_sec(end - start))
    
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(Y_test)
    plt.plot(y_pred)
    plt.xlim((100,150))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y_test)
    plt.plot(y_pred)
    plt.xlim((1000,1080))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.title('NN Model - T2 - Trained on 75%')
    plt.xlabel("Samples - x limits 1000 to 1080 ")
    plt.ylabel("Scaled Resistance")
    
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y_test)
    plt.plot(y_pred)
    plt.xlim((1200,1380))
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y_test)
    plt.plot(y_pred)
    plt.legend(['Train','Predictions'],loc='lower right')
    plt.show()