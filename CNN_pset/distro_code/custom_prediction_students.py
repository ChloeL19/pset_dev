#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 16:20:45 2018

@author: chloeloughridge
"""

# FOR STUDENTS

# this file contains the functions needed for making a prediction on a
# custom image

# import libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np

# function for uploading a custom image
def upload_custom(img_path, new_shape, color, verbose):

    # TODO
    
    return my_img
    

# function for making a prediction on the image
def model_prediction(model, img_path, color, verbose=True, mapping=None):
    new_shape = (model.input_shape[1], model.input_shape[2])
    my_img = upload_custom(img_path, new_shape, color, verbose)
    # if black and white
    if color == True:
        # normalize and feed into model, return index with highest value as output
        out_pred = np.argmax(model.predict(my_img[np.newaxis, :,:] / 255.0))
    elif color == False:
        # normalize and feed into model, return index with highest value as output
        out_pred = np.argmax(model.predict(my_img[np.newaxis, :,:, np.newaxis] / 255.0))
    else:
        print("greyscale value was not valid")
        out_pred = None
    
    if mapping != None:
        pred = mapping[out_pred]
    else:
        pred = out_pred
    
    return pred