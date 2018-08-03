#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 18:24:50 2018

@author: chloeloughridge
"""

# FOR STUDENTS TO COMPLETE
# This is where training, testing, and visualizing the model will take place

# MORE COMFY VERSION

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
# for using tensorboard
from tensorboard_utils import *
import signal, subprocess
# for saving training history
import pickle

# just some housekeeping stuff:
# clear the current keras session so that the graphs will print nicely later
# down the road
keras.backend.clear_session()

# load the training data
# TODO

# load the model architecture so that it is compatible with tensorboard
# TODO
model, tensorboard = None

# compile the model
# TODO

# set up tensorboard for visualization
# suppress the output of the startup function (because it's just messy)
FNULL = open(os.devnull, 'w')
proc = subprocess.Popen(["tensorboard", "--logdir=logs/"], stdout=FNULL, stderr=subprocess.STDOUT)
print("If you'd like to see the training process visualized in tensorboard, paste the following into a browser:")
# copy the correct port to the clipboard, the user will only have to hit command-v
address = 'localhost:6006' #NOTE: THIS MAY BE DIFFERENT FOR C9 ENV
os.system("echo '%s' | pbcopy" % address)
print("It's already copied to clipboard: {}".format(address))

# train the model and log its progress to tensorboard
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