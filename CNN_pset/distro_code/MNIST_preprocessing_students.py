#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 11:08:41 2018

@author: chloeloughridge
"""

#FOR STUDENTS
# This file is for uploading and preprocessing the mnist dataset

# first, let's import keras
import keras
# for visualizing data we need this:
import matplotlib.pyplot as plt
# Keras has the MNIST dataset preloaded. Yay for us!
from keras.datasets import mnist

def preprocess_mnist(verbose=True):
    # import mnist data from keras
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    
    #TODO
    
    # we ultimately want our function to return (X_train, Y_train) and (X_test, Y_test)
    return (X_train, Y_train), (X_test, Y_test)