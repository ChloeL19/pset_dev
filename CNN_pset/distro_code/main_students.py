#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 18:24:50 2018

@author: chloeloughridge
"""

# FOR STUDENTS TO COMPLETE
# This is where training, testing, and visualizing the model will take place

# LESS COMFY VERSION

# import keras library
import keras
import os
# import our helper files
from model_architecture_staff import *
from visualization_staff import *
from custom_prediction_staff import *
from CIFAR_preprocessing_staff import *
from MNIST_preprocessing_staff import *
from EMNIST_preprocessing_staff import *
# for saving training history
import pickle

# load the training data
# TODO

# load the model architecture
# TODO
model = None

# compile the model
# TODO

# train the model
# TODO
train_history = None

# test the model and print a summary of the model
# TODO

# test the model on your own handwriting
# TODO

# visualize the filters and outputs in the model
# TODO

# save the model to a file (along with its training history)
# ask if the user wants to save the model
yes = {'yes','y', 'ye', ''}
no = {'no','n'}
valid = False
# continue asking until user inputs a valid answer
while (valid == False):
    choice = input("Would you like to save your model as a file? ").lower()
    if choice in yes:
        model_file = input("Please enter preferred name of your model file: ")
        #save the model to a filename of your choice
        model.save(model_file) 
        # save the training history to 'trainHistory' file
        with open('./trainHistory', 'wb') as file_pi:
            pickle.dump(train_history.history, file_pi)
        print("The model has been saved as {}".format(model_file))
        valid = True
    elif choice in no:
        print("Okay. Model will not be saved.")
        valid = True
    else:
        print("Please respond with 'yes' or 'no'")
        valid = False