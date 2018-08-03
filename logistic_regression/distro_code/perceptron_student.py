#This is the perceptron STUDENT code --> for students to complete
#Inspiration for the structure of code drawn from here: https://www.kaggle.com/autuanliuyc/logistic-regression-with-tensorflow 

#The libraries we need:
import csv
import numpy as np
import tensorflow as tf

#Objective of this exercise: become familiar with Tensorflow and design an algorithm that predicts the malignancy of breast cancer

#import the data
all_data_list = []

with open('/home/ubuntu/workspace/bc.csv', newline='') as csvfile: #what does the newline thing do?
    myreader = csv.reader(csvfile) 
    for row in myreader:
        #convert each row of the file into an integer array
        row_array = np.array(row).astype(int)
        #add each integer row array to a lsit
        all_data_list.append(row_array)
    #convert the list into a giant array that contains all of the data
    all_data = np.asarray(all_data_list)
    
    #separate the training data from the testing data --> 86% of the data is training data and 14% is testing data
    #generally it's good to split the dataset around the 80%/20% line
    X_train = all_data[:590, 1:10] 
    Y_train = all_data[:590, 11]
    #the testing data
    X_test = all_data[590:683, 1:10] 
    Y_test = all_data[590:683, 11]

#Construction phase

#Set up the TensorFlow placeholders
data = #TODO create a variable for the X inputs (the training parameters)
targets = #TODO create a placeholder for the Y values (the target label values)

#Set up the Tensorflow variables
A = #TODO create a variable for the weights 
b = #TODO create a variable for the bias unit

#initialize the variables in the tensorflow session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#Set up the model
model = #TODO write the model function

#define the optimizer, cost and "goal" of optimization
optimizer = #TODO implement the optimizer and set the learning rate to .003 (this seems to work best, though it is an arbitrary 
#choice and you can come back to experiment with the learning rate later)
cost = #TODO implement the cost function
goal = #TODO write code for minimizing the cost

#Calculate the accuracy of the model
prediction = #TODO write code that will produce the prediction of the model
correct = #TODO create a binary array in which correct predictions by the model are represented by ones
accuracy = #TODO calculate the accuracy of the model 

#state the number of epochs
epochs = 3000 #arbitrary choice
#state the batch size
batch_size = 30 #arbitrary choice

#set up "containers" (python lists) for storing accuracy and cost scores
accuracy_train_record = []
accuracy_test_record = []
cost_record = []

#Execution phase

#TODO create a loop that iterates over the number of epochs
    batch_index = #TODO generate a random set of indices for selecting a batch of input data
    X_train_batch = #TODO select the batch of X values
    Y_train_batch = #TODO select the batch of Y values
    #TODO feed data through the model to optimize the goal
    temp_cost = #TODO store the cost
    temp_train_acc = #TODO feed data through the model to calculate the training accuracy
    temp_test_acc = #TODO feed data through the model to calculate the testing accuracy
    
    #after every 5 epochs, print the loss and the accuracy scores
    if (epoch + 1) % 5 == 0: #the iterative loop starts at epoch = 0, so we shift by one here so that it prints as epoch 1,2,etc.
        print('epoch: {:4d} loss: {:5f} train_acc: {:5f} test_acc: {:5f}'.format(epoch + 1, temp_cost, 
                                                                        temp_train_acc, temp_test_acc))
                                                                        
        