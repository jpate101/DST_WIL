# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 19:03:47 2020

@author: User
"""
#https://stackoverflow.com/questions/51306862/how-do-i-use-tensorflow-gpu

import tensorflow as tf
import keras
import sklearn

print(tf.version.VERSION)

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

print(tf.test.is_gpu_available())

#print('The scikit-learn version is {}.'.format(sklearn.__version__))

#from tensorflow.python.client import device_lib 
#print(device_lib.list_local_devices())