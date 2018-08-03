#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 11:56:47 2018

@author: chloeloughridge
"""

# FOR STUDENTS
# This file is for uploading and preprocessing the cifar10 dataset

# first, let's import keras
import keras
# for visualizing data we need this:
import matplotlib.pyplot as plt
# for working with matrices of numbers
import numpy as np
# Keras has the cifar10 dataset preloaded. Yay for us!
from keras.datasets import cifar10

# a function for preprocessing the cifar10 dataset
def preprocess_cifar(verbose=True):
    # import cifar10 data from keras
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    
    # if in verbose mode, visualize a training image
    # TODO
    
    # normalize the X data before feeding it into our model
    # TODO
    
    # we also need to convert the Y data into one-hot vectors
    # TODO
    
    # for interpretation purposes, create a dictionary mapping the label numbers
    # to image labels
    # information from here: https://www.cs.toronto.edu/~kriz/cifar.html
    num_to_img = {
        0:"airplane",
        1:"automobile",
        2:"bird",
        3:"cat",
        4:"deer",
        5:"dog",
        6:"frog",
        7:"horse",
        8:"ship",
        9:"truck"
    }
    
    # if in verbose mode, print the label of the image displayed above
    # TODO
    
    return (X_train, Y_train), (X_test, Y_test), num_to_img