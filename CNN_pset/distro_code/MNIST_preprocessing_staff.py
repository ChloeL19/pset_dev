#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 11:08:41 2018

@author: chloeloughridge
"""

#STAFF VERSION

# first, let's import keras
import keras
# for visualizing data we need this:
import matplotlib.pyplot as plt
# Keras has the MNIST dataset preloaded. Yay for us!
from keras.datasets import mnist

def preprocess_mnist(verbose=True):
    # import mnist data from keras
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    
    if verbose == True:
        # It's always a good idea to first visualize the data
        # Let's take a look at the shape of each image in the X_train dataset
        print("The shape of each image: {}".format(X_train[0].shape))
        # Now let's plot one of the images in the training set
        print("An example training image:")
        plt.imshow(X_train[0])
        plt.show()
        
        print("Label: {}".format(Y_train[0]))
        
    # We need to reshape the X_train and X_test data so that each image within it is a 28 by 28 pixel square.
    # Like this: [the number of training examples, image width, image height, the number of color channels]
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    
    
    # We need to normalize the X data before feeding into our model
    # This means we want to put each pixel value in the range 0 to 1
    X_train = X_train/255 # divide by max value of pixel values to put in range 0 to 1
    X_test = X_test/255
    
    # We also need to convert the Y data into one-hot vectors.
    num_classes = 10 # ten output classes because our data represents the digits from 0 to 9
    Y_train = keras.utils.to_categorical(Y_train, num_classes) # pset write up explains what one-hot vectors are
    Y_test = keras.utils.to_categorical(Y_test, num_classes)
    
    # we ultimately want our function to return (X_train, Y_train) and (X_test, Y_test)
    return (X_train, Y_train), (X_test, Y_test)