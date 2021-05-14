# -*- coding: utf-8 -*-
"""
Created on Wed May 12 03:18:59 2021

@author: AJ
"""
# Importing different packages that are required
import numpy as np
import pickle
import kerastuner as kt
import tensorflow as tf
from sklearn.model_selection import train_test_split

# opening pkl file that is already created
f = open('Dataset.pkl','rb')
dataset = pickle.load(f)     
f.close()

"""separating the dataset into features and targets
Reshaping the target to form a 2-D array
"""
features = dataset[:,:-1]
target = np.reshape(dataset[:,-1],(len(dataset),1))

# Split data into training and test set 
x_train, x_test, y_train, y_test = train_test_split(features, target ,test_size = 0.2 , random_state = 0)

"""
building a neural network with variable number of hidden layers ranging from 2 to 4
The hyperparameters are tuned using keras tuner
Varying number of neurons from 5 to 50 with steps of 5
L2 regulariser is used to prevent overfitting
A choice is given for the optimizer between Adam, Adagrad and Adadelta
Mean_squared_error loss function is being used
"""


def build_model(hp):
    
    model = tf.keras.Sequential()
    
    for i in range(hp.Int('n_layers',2,4)):
    
        model.add(tf.keras.layers.Dense(units=hp.Int(f'{i}_units', min_value=5, max_value=50, step=5),
                                        activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    
    model.add(tf.keras.layers.Dense(units=(np.shape(target)[1])))
    model.compile(optimizer= hp.Choice('optimiser',['adam','adagrad','adadelta']),loss='mean_squared_error')
    
    return model

# Performing random search with objective of minimising loss function
tuner = kt.RandomSearch(build_model,
                     objective='val_loss',
                     max_trials=5,
                     executions_per_trial=3,
                     directory='optimization_folder')
tuner.search_space_summary()
tuner.search(x_train, y_train, validation_data=(x_test,y_test), epochs=10)


# finding the best hyperparameters for the neural network
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]


# Building the model with the optimised hyperparameters
best_model = tuner.hypermodel.build(best_hps)


""" Call back is performed when the difference between the loss function of 
    traing data and test data is large. This also prevents overfitting
    """
monitor = tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=1e-3,patience=5)
history = best_model.fit(x_train, y_train, validation_data=(x_test,y_test), callbacks=[monitor], epochs=100)
