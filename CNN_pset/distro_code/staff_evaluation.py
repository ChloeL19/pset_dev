#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:26:11 2018

@author: chloeloughridge
"""

# Some ideas for how to go about evaluating student submissions
# This file is by no means complete

import pickle

# accuracy threshold values
mnist_thresh = 0.99
emnist_thresh = 0.85
cifar_thresh = 0.75

# check to see if the highest accuracy the model ever achieved during training
# is higher than the threshold

# open the training history pickle file
with open('./trainHistory', 'rb') as f:
    history = pickle.load(f)

# identify the highest accuracy score 
max_val = 0
for acc_score in history['val_acc']:
    if acc_score > max_val:
        max_val = acc_score

# we'll need to somehow identify which dataset the model is designed for
if max_val >= mnist_thresh:
    print("Yay! The model passed the MNIST threshold!")
else:
    print("Oh no, the model did not pass the MNIST threshold")