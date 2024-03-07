# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:31:55 2021

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

from sklearn.datasets import load_iris
from sklearn import tree


if __name__=="__main__":
    
    """
    #load
    with open('Models/model_pickle','rb') as f:
        DT = pickle.load(f)
        
        tree.plot_tree(DT, max_depth=2)  
     
    """
    clf = tree.DecisionTreeClassifier(random_state=0)
    iris = load_iris()
    
    clf = clf.fit(iris.data, iris.target)
    tree.plot_tree(clf)  
  
        
        
        
        
        
        
        
        
        
        
        
        
        