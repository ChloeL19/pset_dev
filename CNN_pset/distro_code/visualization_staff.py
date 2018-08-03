#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:38:20 2018

@author: chloeloughridge
"""

# FOR STAFF ONLY

# first, import libraries
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to black/white representation
    x *= 255
    #x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# The function for visualizing the filters of the model
def vis_filter(model, filter_index, layer_name):
    
    # create a dictionary of the model's layers
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    img_width = model.input_shape[1]
    img_height = model.input_shape[2]
    channels = model.input_shape[3]
    
    # create a loss function that we will use to maximize the activation of the specified layer
    layer_output = layer_dict[layer_name].output # accesing the output of the specified layer stored in our dictionary
    loss = K.mean(layer_output[:,:,:, filter_index]) # averaging all the outputs of the filter --> remember that the filter
                                                    # is a 2-D "square" --> the middle two numbers represent height/width
                                                    # of that square, the first number represents batch size, and the 
                                                    # final number represents the number of filters (we're accessing a
                                                    # specific one here)
    # compute the gradient of the input picture with respect to this loss.
    # this means we'll be updating the pixels of the input image, not the weights of the network --> clever!
    grads = K.gradients(loss, model.input)[0] # I don't know what the [0] means at the end of this line
    
    # normalizing the gradient --> we don't want the magnitude of our gradient ascent/descent step to be infuenced heavily 
    # by the gradient --> the gradient gives us the direction we want to take --> so normalizing helps the optimization
    # algorithm perform better
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    
    iterate = K.function([model.input], [loss, grads]) # this is a fancy way of writing a custom function that takes
                                                    # an input image as an input, and returns the loss and gradients
                                                    # for that image
    
    # running gradient ascent to maximize the activations of the filter
    
    # generating a grey test image
    input_img = np.random.random((1, img_width, img_height, channels)) * 20 + 128.
    
    # Choosing an arbitrary value for "step" --> pretty sure this is like the learning rate
    step = 5
    
    # 20 steps of gradient ascent
    for i in range(50):
        loss_value, grads_value = iterate([input_img])
        input_img += grads_value * step
        
    
    # creating the image
    img = input_img
    img = deprocess_image(img)
    
    print("Image of what filter number {} from your model is 'seeing': ".format(filter_index))
    
    # displaying with matplotlib approach
    plt.imshow(img.squeeze())
    plt.show()
    
# the function for visualizing the outputs of the model
def vis_output(model, output_index):
      
    img_width = model.input_shape[1]
    img_height = model.input_shape[2]
    channels = model.input_shape[3]
    
    loss = K.mean(model.output[:, output_index]) # averaging all the outputs of the filter --> remember that the filter
                                                    # is a 2-D "square" --> the middle two numbers represent height/width
                                                    # of that square, the first number represents batch size, and the 
                                                    # final number represents the number of filters (we're accessing a
                                                    # specific one here)
    # compute the gradient of the input picture with respect to this loss.
    # this means we'll be updating the pixels of the input image, not the weights of the network --> clever!
    grads = K.gradients(loss, model.input)[0] # I don't know what the [0] means at the end of this line
    
    # normalizing the gradient --> we don't want the magnitude of our gradient ascent/descent step to be infuenced heavily 
    # by the gradient --> the gradient gives us the direction we want to take --> so normalizing helps the optimization
    # algorithm perform better
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    
    iterate = K.function([model.input], [loss, grads]) # this is a fancy way of writing a custom function that takes
                                                    # an input image as an input, and returns the loss and gradients
                                                    # for that image
            
    
    # running gradient ascent to maximize the activations of the filter
    
    # generating a grey test image
    input_img = np.random.random((1, img_width, img_height, channels)) * 20 + 128.
    
    # Choosing an arbitrary value for "step" --> pretty sure this is like the learning rate
    step = 5
    
    # 20 steps of gradient ascent
    for i in range(50):
        loss_value, grads_value = iterate([input_img])
        input_img += grads_value * step
        
    # converting to image
    img = input_img
    img = deprocess_image(img)
    
    print("Your model's visualization of what output node {} represents: ".format(output_index))
    
    # display the image
    plt.imshow(img.squeeze())
    plt.show()



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    