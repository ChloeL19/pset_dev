#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 12:15:52 2018

@author: chloeloughridge
"""

# FOR STUDENTS
# MORE COMFY VERSION

# This is a file for constructing the model's architecture and its tensorboard object

# import our libraries
import keras
from keras.layers import Input, Conv2D, Dense, Activation, Flatten, Dropout, MaxPooling2D, ZeroPadding2D
from keras.models import Model
from keras.callbacks import TensorBoard

# function for constructing the model's architecture
def create_model(pic_width, pic_height, color_channels, num_classes, run_version):
    
   
    
    #TODO: DESIGN MODEL'S ARCHITECTURE
    
    
    
    # create a tensorboard object for visualization during training
    tensorboard = TensorBoard(log_dir="logs/run_{}".format(run_version), write_graph=True)
    
    # return the model and the tensorboard object
    return model, tensorboard

# for testing and debugging: if you run this file from the command line, the following will execute
if __name__ == "__main__":
    model = create_model(28, 28, 1, 10)
    print(model.summary())