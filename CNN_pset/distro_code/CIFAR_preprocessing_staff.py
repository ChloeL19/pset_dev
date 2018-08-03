#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 11:56:47 2018

@author: chloeloughridge
"""

# FOR STAFF

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
    
    if verbose == True:
        # visualize an image
        print("The shape of each image: {}".format(X_train[0].shape))
        print("An example training image:")
        plt.imshow(X_train[0])
        plt.show()
    
    # we need to normalize the X data before feeding into our model
    X_train = X_train/255 # divide by max value of pixel values to put in range 0 to 1
    X_test = X_test/255
    
    # we also need to convert the Y data into one-hot vectors
    num_classes = 10
    Y_train = keras.utils.to_categorical(Y_train, num_classes) # may want to explain this more clearly [background]
    Y_test = keras.utils.to_categorical(Y_test, num_classes)
    
    # for interpretation purposes, create a dictionary mapping the label numbers
    # to image labels
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
    
    if verbose == True:
        print("Label: {}".format(num_to_img[np.argmax(Y_train[0])]))
    
    return (X_train, Y_train), (X_test, Y_test), num_to_img