# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 19:59:03 2020

@author: User
"""
import numpy as np

a = [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]

a = a[:5]
print(a)

test = [[12.5,9.80665,1025.9,1.18831,10.0,1.226,14.4,0,0,0.500000,1.000000,0.300000,1,.5,.5,.5,0,0,0,0,0,0],[1,9.80665,1025.9,1.18831,10.0,1.226,14.4,0,0,0.500000,1.000000,0.300000,1,.5,.5,.5,0,0,0,0,0,0],[25,9.80665,1025.9,1.18831,10.0,1.226,14.4,0,0,0.500000,1.000000,0.300000,1,.5,.5,.5,0,0,0,0,0,0]]

test = np.array(test)

test = np.reshape(test,(test.shape[0],test.shape[1],1))

predictions = model.predict(test) 
print(predictions)

#import datetime
#a = datetime.datetime.now()
#b = datetime.datetime.now()
#c = b - a