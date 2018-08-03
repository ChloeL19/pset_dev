#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 12:15:52 2018

@author: chloeloughridge
"""

# FOR STAFF
# THE LESS COMFY VERSION

# import our libraries
import keras
from keras.layers import Input, Conv2D, Dense, Activation, Flatten, Dropout, MaxPooling2D, ZeroPadding2D
from keras.models import Model

# function for constructing the model's architecture
def create_model(pic_width, pic_height, color_channels, num_classes):
    
    input_shape = [pic_width, pic_height, color_channels]
    X_input = Input(input_shape)
    
    # THIS IS WHERE MODEL ARCHITECTURE DESIGN STARTS
    # The following is a super simple model with one convolutional layer
    
    X = Conv2D(32, kernel_size=(3,3), strides=2)(X_input)
    X = Activation('relu')(X)
    
    X = Flatten()(X)
    X = Dense(num_classes, activation='softmax')(X)
    
    # instantiate the model
    model = Model(inputs=X_input, outputs=X)
    
    # return the model
    return model

if __name__ == "__main__":
    model = create_model(28, 28, 1, 10)
    print(model.summary())