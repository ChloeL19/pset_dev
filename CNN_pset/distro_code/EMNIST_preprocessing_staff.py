#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 11:26:59 2018

@author: chloeloughridge
"""

# STAFF CODE
# keras doesn't have this dataset loaded, so we wrote some upload code for students

# our libraries
import keras
from scipy.io import loadmat
import numpy as np
# for visualizing data
import matplotlib.pyplot as plt

# a function for preprocessing EMNIST data
def preprocess_emnist(file_path="../EMNIST_data/matlab/emnist-balanced", verbose = True):

    # a function for loading the emnist data
    # credits: https://www.kaggle.com/marcose18/cnn-on-emnist-dataset
    def load_data(mat_file_path, width=28, height=28, max_=None, verbose=False):
        def rotate(img):
            # Used to rotate images (for some reason they are transposed on read-in)
            flipped = np.fliplr(img)
            return np.rot90(flipped)
    
        def display(img, threshold=0.5):
            # Debugging only
            render = ''
            for row in img:
                for col in row:
                    if col > threshold:
                        render += '@'
                    else:
                        render += '.'
                render += '\n'
            return render
    
        mat = loadmat(mat_file_path)
        # Load convoluted list structure form loadmat
        mat = loadmat(mat_file_path)
    
        # Load char mapping
        mapping = {kv[0]: kv[1:][0] for kv in mat['dataset'][0][0][2]}
    
        # Load training data
        if max_ == None:
            max_ = len(mat['dataset'][0][0][0][0][0][0])
        training_images = mat['dataset'][0][0][0][0][0][0][:max_].reshape(
            max_, height, width, 1)
        training_labels = mat['dataset'][0][0][0][0][0][1][:max_]
    
        # Load testing data
        if max_ == None:
            max_ = len(mat['dataset'][0][0][1][0][0][0])
        else:
            max_ = int(max_ / 6)
        testing_images = mat['dataset'][0][0][1][0][0][0][:max_].reshape(
            max_, height, width, 1)
        testing_labels = mat['dataset'][0][0][1][0][0][1][:max_]
    
        # Reshape training data to be valid
        if verbose == True:
            _len = len(training_images)
        for i in range(len(training_images)):
            if verbose == True:
                print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100))
            training_images[i] = rotate(training_images[i])
        if verbose == True:
            print('')
    
        # Reshape testing data to be valid
        if verbose == True:
            _len = len(testing_images)
        for i in range(len(testing_images)):
            if verbose == True:
                print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100))
            testing_images[i] = rotate(testing_images[i])
        if verbose == True:
            print('')
        
        # Convert type to float32
        training_images = training_images.astype('float32')
        testing_images = testing_images.astype('float32')
    
        # Normalize to prevent issues with model
        training_images /= 255
        testing_images /= 255
    
        nb_classes = len(mapping)
        
        
    
        return ((training_images, training_labels), (testing_images, testing_labels), mapping, nb_classes)
    
    # load the data from EMNIST 
    (X_train, Y_train), (X_test, Y_test), mapping, nb_classes = load_data(file_path)
    
    # convert Y labels into one-hot vectors 
    Y_train = keras.utils.to_categorical(np.squeeze(Y_train), nb_classes) 
    Y_test = keras.utils.to_categorical(np.squeeze(Y_test), nb_classes)
    
    if verbose == True:
        # visualize the dataset
        print("Shape of each image in the training set: {}".format(X_train[0].shape))
        print("An example image from the dataset:")
        plt.imshow(X_train[3].squeeze())
        plt.show()
        
        num = np.argmax(Y_train[3])
        mapped = mapping[num]
        print("Label: {}".format(chr(mapped)))
    
    return (X_train, Y_train), (X_test, Y_test), mapping
    
    