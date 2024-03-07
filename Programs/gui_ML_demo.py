# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 08:17:10 2021

@author: User
"""

from tkinter import *
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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import time
import datetime


if __name__=="__main__":
    
    
    
    root = Tk()
    
    def graph():
        pass
    
    def hide_all_frames():
        pass
    
    def myclick_DT():
        print("test DT ")
                
        for widget in bottomframe.winfo_children():
            widget.destroy()
        
        
        
        Displaced_V = e_1.get()
        Length = e_2.get()
        Draft = e_3.get()
        Trim = e_4.get()
        
        Offsets = e_5.get()
        Offsets = Offsets.split(",")
        Offsets = [float(Offsets[0]),float(Offsets[1]),float(Offsets[2]),float(Offsets[3])]
        
        speed = e_6.get()
        
        data = [speed,Length,Draft, Displaced_V,1,Offsets[1],Offsets[2],Offsets[3],5,Trim]
        data = np.array(data)
        print(data)
        
        speed_count = 1
        speed_data = []
        for i in range(1,21):
            speed_data.append([i,Length,Draft, Displaced_V,1,Offsets[1],Offsets[2],Offsets[3],5,Trim])
       
        speed_data = np.array(speed_data)
        
        
        with open('Models/model_pickle','rb') as f:
            DT = pickle.load(f)
            
        pred = DT.predict(data.reshape(1,-1))
        speed_pred = DT.predict(speed_data)
        
        print(pred)
        
        
        myLabel = Label(bottomframe,text=pred)
        myLabel.pack()

        #visualise
        plt.Figure(figsize=(16,8))
        plt.plot(speed_pred)
        plt.show()
    
        figure = plt.Figure(figsize = (8,4),dpi = 100)
        figure.add_subplot(111).plot(speed_pred)
        
        chart = FigureCanvasTkAgg(figure, bottomframe)
        chart.get_tk_widget().pack()
    
    
        pass
    
    def myclick_NN():
        print("test NN")
        
        for widget in bottomframe.winfo_children():
            widget.destroy()
        
        Displaced_V = e_1.get()
        Length = e_2.get()
        Draft = e_3.get()
        Trim = e_4.get()
        
        Offsets = e_5.get()
        Offsets = Offsets.split(",")
        Offsets = [float(Offsets[0]),float(Offsets[1]),float(Offsets[2]),float(Offsets[3])]
        
        speed = e_6.get()
        
        data = [speed,Length,Draft, Displaced_V,1,Offsets[1],Offsets[2],Offsets[3],5,Trim]
        data = np.array(data)
        print(data)
        
        speed_count = 1
        speed_data = []
        for i in range(1,21):
            speed_data.append([i,Length,Draft, Displaced_V,1,Offsets[1],Offsets[2],Offsets[3],5,Trim])
       
        speed_data = np.array(speed_data)
        
        
        with open('Models/model_pickle_2','rb') as f:
            NN = pickle.load(f)
            
        pred = NN.predict(data.reshape(1,-1))
        speed_pred = NN.predict(speed_data)
        
        print(pred)
        myLabel = Label(bottomframe,text=pred)
        myLabel.pack()
        
        #print(speed_data)
        #print(speed_pred)
        
        #visualise
        plt.figure(figsize=(16,8))
        plt.plot(speed_pred)
        plt.show()
        
        figure = plt.Figure(figsize = (8,4),dpi = 100)
        figure.add_subplot(111).plot(speed_pred)
        
        chart = FigureCanvasTkAgg(figure, bottomframe)
        chart.get_tk_widget().pack()
        
        pass
    
    topframe = Frame(root)
    bottomframe = Frame(root)
    midframe = Frame(root)
    topframe.pack(side=TOP)
    midframe.pack()
    bottomframe.pack(side=BOTTOM)
    
    button_DT = Button(midframe, text="Submit DT",fg="green",command=myclick_DT)
    button_DT.pack(side=LEFT)
    button_NN = Button(midframe, text="Submit NN",fg="blue",command=myclick_NN)
    button_NN.pack(side=RIGHT)
    
    
    L_1 = Label(topframe, text="Displaced volume of hull form in cubic meters [5000 - 6500]")
    L_1.pack()
    e_1 = Entry(topframe,width = 50,borderwidth=5)
    e_1.pack()
    
    L_2 = Label(topframe, text="Length of hull in meters [110 - 150]")
    L_2.pack()
    e_2 = Entry(topframe,width = 50,borderwidth=5)
    e_2.pack()
    
    L_3 = Label(topframe, text="draft of hull in meters  [4.3 - 5.8]")
    L_3.pack()
    e_3 = Entry(topframe,width = 50,borderwidth=5)
    e_3.pack()
    
    L_4 = Label(topframe, text="Trim of hull in degrees  [-3 - 6]")
    L_4.pack()
    e_4 = Entry(topframe,width = 50,borderwidth=5)
    e_4.pack()
    
    L_5 = Label(topframe, text="Michlet offsets  [M1]")
    L_5.pack()
    e_5 = Entry(topframe,width = 50,borderwidth=5)
    e_5.pack()
    
    L_6 = Label(topframe, text="Speed in m/s [0 - 20]")
    L_6.pack()
    e_6 = Entry(topframe,width = 50,borderwidth=5)
    e_6.pack()
    
    root.mainloop()