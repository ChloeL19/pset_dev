#This is the perceptron STAFF SOLUTION code
#Inspiration for the structure of code drawn from here: https://www.kaggle.com/autuanliuyc/logistic-regression-with-tensorflow

#The libraries we need:
import csv
import numpy as np
import tensorflow as tf

#Objective of this exercise: become familiar with Tensorflow and predict the malignancy of breast cancer 

#import the data 
all_data_list = []

with open('/home/ubuntu/workspace/bc.csv', newline='') as csvfile:
    myreader = csv.reader(csvfile) 
    for row in myreader:
        #convert each row of the file into an integer array
        row_array = np.array(row).astype(int)
        #add each integer row array to a lsit
        all_data_list.append(row_array)
    #convert the list into a giant array that contains all of the data
    all_data = np.asarray(all_data_list)
    
    #separate the X and Y training data from the testing data
    X_train = all_data[:590, 1:10]  
    Y_train = all_data[:590, 11]
    #the testing data
    X_test = all_data[590:683, 1:10] 
    Y_test = all_data[590:683, 11]

#Set up the Tensorflow placeholders
data = tf.placeholder(tf.float32, shape=[None, 9]) #any number of training examples can be passed in, there will be 9 x parameters
targets = tf.placeholder(tf.float32, shape=[None, 1])

#Set up the Tensorflow variables
A = tf.Variable(tf.random_normal(shape=[9,1])) #the "weights" 
b = tf.Variable(tf.random_normal(shape=[1,1])) #the bias unit, which is just a random float
#initialize the variables in the tensorflow session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#Set up the model that will be used 
model = tf.matmul(data, A) + b 

#define the optimizer and the cost and the "goal" of optimization
#we'll use a gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=.003) #this is an adjustable hyperparameter
#we'll use the cross-entropy loss function
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model, labels=targets))
#the goal is to reduce the cost
goal = optimizer.minimize(cost)

#define the accuracy
#make a prediction
prediction = tf.round(tf.sigmoid(model)) #NOTE: OHHHH students can use their own sigmoid functions here!
#figure out how many of the predictions are correct (vectorize it)
correct = tf.cast(tf.equal(prediction, targets), dtype = tf.float32) #cast converts the boolean values in the array to floats
#sum up performance with a number: ratio of correct predictions to total number of predictions
accuracy = tf.reduce_mean(correct)

#state the number of epochs
epochs = 3000 #arbitrary choice
#state the batch size
batch_size = 30 #arbitrary choice

#set up the recording "containers" for accuracy and cost scores
accuracy_train_record = []
accuracy_test_record = []
cost_record = []

#iterate over the epochs
for epoch in range(epochs): #check syntax in for-loop
    #select a random batch to feed into the model --> remember both X and Y components
    batch_index = np.random.choice(len(X_train), size=batch_size) #generating a random set of indices
    X_train_batch = X_train[batch_index]
    Y_train_batch = np.matrix(Y_train[batch_index]).T #we must transpose because of the dot product in the model
    #feed the data through the model to optimize it
    sess.run(goal, feed_dict={data: X_train_batch, targets: Y_train_batch})
    #keep track of the loss
    temp_cost = sess.run(cost, feed_dict={data: X_train_batch, targets: Y_train_batch})
    #calculate the accuracy on the training dataset by feeding the training dataset through the accuracy "pipe system"
    temp_train_acc = sess.run(accuracy, feed_dict={data: X_train, targets: np.matrix(Y_train).T})
    #calculate the accuracy on the test dataset by feeding the test dataset through the accuracy "pipe system"
    temp_test_acc = sess.run(accuracy, feed_dict={data: X_test, targets: np.matrix(Y_test).T})
    #after every 5 epochs, print the loss and the accuracy scores
    if (epoch + 1) % 5 == 0: #the iterative loop starts at epoch = 0, so shifting by one here
        print('epoch: {:4d} loss: {:5f} train_acc: {:5f} test_acc: {:5f}'.format(epoch + 1, temp_cost, 
                                                                        temp_train_acc, temp_test_acc))
                                                                        
        