#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 13:56:04 2018

@author: chloeloughridge
"""

# The file where training, testing, and visualizing the model takes place
# perhaps this can become staff example implementation code later on

# LESS COMFY VERSION

# for starting within ipython console: exec(open("main.py").read())

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

# welcome the user
print("Welcome to the epic convolutional neural network training and testing ground! (Aka 'main.py')")
print("This will later become staff example code.")
print("Press ctrl-C at any time to exit in the middle of the program")

# for future questions
yes = {'yes','y', 'ye', ''}
no = {'no','n'}

# loading the training data
# ask the user about which dataset to train on
val_resp = False
while val_resp == False:
    dataset = input("Which dataset would you like to train your model on? [mnist, emnist, or cifar] ").lower()
    if dataset == 'mnist':
        print("Now loading the mnist training data . . . ")
        (X_train, Y_train), (X_test, Y_test) = preprocess_mnist()
        val_resp = True
    elif dataset == 'emnist':
        print("Now loading the emnist training data . . . ")
        (X_train, Y_train), (X_test, Y_test), returned_map = preprocess_emnist()
        val_resp = True
    elif dataset == 'cifar':
        print("Now loading the cifar training data . . . ")
        (X_train, Y_train), (X_test, Y_test), returned_map = preprocess_cifar()
        val_resp = True
    else:
        print("Please enter a valid dataset name.")

# wait for user
input("Data loaded. Press Enter to Continue . . . ")

# creating the model's architecture
print("Loading the model's architecture . . . ")
model = create_model(X_train.shape[1], X_train.shape[2], X_train.shape[3], Y_train.shape[1])

# wait for user
input("Model architecture loaded. Press Enter to continue . . . ")

# compile the model
print("Compiling the model . . . ")
model.compile(loss=keras.losses.categorical_crossentropy, #
             optimizer=keras.optimizers.Adam(), #students can choose optimizers 
             metrics=['accuracy'])

# wait for user
input("Model compiled. Press Enter to continue . . . ")

# train the model
print("Training the model . . . ")
# save the training history of the model
batch_size = 128
epochs = 10
train_history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, Y_test))
# ask if the user wants to train the model again
val_resp = False
while val_resp == False:
    train_again = input("Would you like to train the model again? [y/n] ").lower()
    if train_again in yes:
        print("Training the model again . . .")
        train_history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, Y_test))
        val_resp = False
    elif train_again in no:
        val_resp = True
    else:
        print("please answer 'yes' or 'no'")
        val_resp = False

# wait for user
input("Model has been trained. Press Enter to continue . . . ")

# print summary of model and confirm model evaluation
print("Confirming the model evaluation . . . ")
final_acc = model.evaluate(X_test, Y_test)
print("The final loss and accuracy: {}".format(final_acc))
print("Here's the model summary:")
model.summary()

# wait for user
input("Press Enter to continue . . . ")

# ask if the user wants to upload a custom image to the model
val = False
while val == False:
    resp = input("Would you like to upload a(nother) custom image to the model? ")
    if resp in yes:
        valid_resp = False
        while valid_resp == False:
            img_path = input("Please enter the path of your image: ")
            # if this is a valid path
            if os.path.exists(img_path):

                # tweak parameters based on dataset
                if dataset == 'mnist':
                    color = False
                    mapping = None
                    data_resp = True
                elif dataset == 'emnist':
                    color = False
                    mapping = returned_map
                    data_resp = True
                elif dataset == 'cifar':
                    color = True
                    mapping = returned_map
                    data_resp = True

                # make a model prediction!
                prediction = model_prediction(model, img_path, color, mapping=mapping)
                # print the prediction
                print("Model prediction: {}".format(prediction))
                
                valid_resp = True
            else:
                print("Please input a valid file path.")
                valid_resp = False
        val = False
    elif resp in no:
        print("Okay. Moving on . . . ")
        val = True
    else:
        print("Please respond with 'yes' or 'no'")
        valid = False

# ask if the user wants to see some of the model's filters
val = False
while val == False:
    resp = input("Would you like to dive deeper and visualize a(nother) model filter? ")
    if resp in yes:
        # dictionary of model's layers
        layer_dict = dict([(layer.name, layer) for layer in model.layers])
        
        valid_layer = False
        while valid_layer == False:
            layer_name = input("Choose which convolutional layer of your model to visualize: ")
            if layer_name in layer_dict:
                val_ind = False
                while val_ind == False:
                    max_ind = layer_dict[layer_name].output_shape[3]
                    ind = int(input("Please enter an index number less than {}: ".format(max_ind)))
                    if ind < max_ind:
                        print("Visualizing the filter . . . ")
                        vis_filter(model, ind, layer_name)
                        val_ind = True
                    else:
                        val_ind = False
                valid_layer = True
            else:
                valid_layer = False
        val = False
    elif resp in no:
        print("Okay.")
        val = True
    else:
        print("Please respond with 'yes' or 'no'")
        valid = False
        
# ask if the user wants to see one of the model's outputs
val = False
while val == False:
    resp = input("Would you like to visualize an(other) output node? ")
    if resp in yes:
        valid_node = False
        while valid_node == False:
            node_name = int(input("Choose which output node to visualize: "))
            num_classes = Y_train.shape[1]
            if node_name < num_classes:
                # visualize that node
                vis_output(model, node_name)     
                valid_node = True
            else:
                valid_node = False
        val = False
    elif resp in no:
        print("Okay. Moving on . . . ")
        val = True
    else:
        print("Please respond with 'yes' or 'no'")
        valid = False
        
# ask if the user wants to save the model
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
   












