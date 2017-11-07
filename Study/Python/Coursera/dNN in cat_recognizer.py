# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 14:12:11 2017

@author: Surface
"""

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v2 import *

%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

%load_ext autoreload
%autoreload 2

np.random.seed(1)


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

# 2 layer NN

n_0 = 12288     # num_px * num_px * 3
n_1 = 7
n_2 = 1
layers_dims = (n_0, n_1, n_2)

def two_layer_model(X,Y,layer_dims,learning_rate = 0.0075,num_iteration = 3000,print_cost = False):
    np.random.seed(1)
    grads = {}
    costs = []
    m = X.shape[1]
    (n_1,n_2,n_3) = layer_dims
    
    parameters = initialize_parameters(n_1,n_2,n_3)
    
    W1 = parameters["W1"]
    b1 = parameters["b2"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    for i in range(0,num_iterations):
        #forward propagation
        A1,cache1 = linear_activation_forward(X,W1,b1,activation = "relu")
        A2,cache2 = linear_activation_forward(A1,W2,b2,activation = "sigmoid")
        #compute cost
        cost = compute_cost(A2,Y)
        
        #backprop  
        dA2 = -(np.divide(Y,A2) - np.divide(1 - Y,1 - A2))
        
        dA1,dW2,db2 = linear_activation_backward(dA2,cache2,activation = "sigmoid")
        dA0,dW1,db1 = linear_activation_backward(dA1, cache1,activation = "relu")
        #compute completed
        
        grads["dW1"] = dW1
        grads["db1"] = db1
        grads["dW2"] = dW2
        grads["db2"] = db2
        
        parameters = update_parameters(parameters,grads,learin_rate)
        
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}:{}".format(i,np.squeeze(cost)))
            costs.append(cost)
            
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

def L_layer_model(X,Y,layer_dims,learning_rate = 0.0075,num_iterations = 3000, print_cost = False):
    
    np.random.seed(1)
    cost = []
    parameters = initialize_parameters_deep(layers_dims)
    
    for i in range(0,num_iterations):
        
        AL,caches = L_model_forward(X,parameters)
        
        cost = compute_cost(AL,Y)
        
        grads = L_model_backward(AL,Y,caches)
        
        parameters = update_parameters(parameters,grads,learning_rate)
        
        if print_cost and 1 % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)
    
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

