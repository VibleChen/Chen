# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 09:32:53 2017

@author: Surface
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planer_utils import plot_decision_boundary, sigmoid,load_planar_dataset, load_extra_datasets

%matplotlib inline

X,Y = load_planar_dataset()

shape_X = X.shape
shape_Y = Y.shape
m = X.shape[1]

def layer_sizes(X,Y):
    n_0 = X.shape[0]
    n_1 = 4
    n_2 = Y.shape[0]
    return (n_0,n_1,n_2)

def initialize_parameters(n_0,n_1,n_2):
    
    np.random.seed(2)
    W1 = np.random.randn((n_1,n_0)) * 0.01
    b1 = np.zeros((n_1,1))
    W2 = np.random.randn((n_2,n_1)) * 0.01
    b2 = np.zeros((n_2,1))
    
    parameters = {"W1" : W1
                  "b1" : b1
                  "W2" : W2
                  "b2" : b2}
    
    return parameters

def forward_propagate(X,parameters):
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    Z1 = np.dot(W1,X) + b1
    A1 = tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    
    cache = {"Z1" : Z1
             "A1" : A1
             "Z2" : Z2
             "A2" : A2}
    
    return A2,cache

def compute_cost(A2,Y,parameters):
    
    m = Y.shape[1]
    logprobs = np.mutiply(Y,np.log(A2)) + np.mutiply((1 - Y),np.log(1 - A2))
    cost = -np.sum(logprobs) / m
    
    return cost

def backward_propagation(parameters,cache,X,Y):
    m = Y.shape[1]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2,A1.T) / m
    db2 = np.sum(dZ2,axis = 1,keepdims = True) / m
    dZ1 = np.dot(W2.T,dZ2) * (1 - A1 **2)
    dW1 = np.dot(dZ1,X.T) / m
    db1 = np.sum(dZ1,axis = 1,keepdims = True) / m
    
    grads = {"dW1" : dW1
             "db1" : db1
             "dW2" : dW2
             "db2" : db2}
    
    return grads

def update_parameters(parameters,grads,learning_rate = 1.2):

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - leaining_rate * db2
    
    parameters = {"W1" : W1
                  "b1" : b1
                  "W2" : W2
                  "b2" : b2}
    
    return parameters

def nn_model(X,Y,num_iteration = 10000,print_cost = False):
    
    np.random.seed(3)
    n_0 = layer_sizes(X,Y)[0]
    n_1 = layer_sizes(X,Y)[1]
    n_2 = layer_sizes(X,Y)[2]
    
    parameters = initialize_parameters(n_1,n_2,n_3)
    #W1 = parameters["W1"]
    #b1 = parameters["b1"]
    #W2 = parameters["W2"]
    #b2 = parameters["b2"]
    
    for i in range (num_iteration):
        A2,cache = forward_propagetion(X,parameters)
        cost = compute_cost(A2,Y,parameters)
        grads = backward_propagation(parameters,cache,X,Y)
        parameters = update_parameters(parameters,grads,learning_rate = 1.2)
        
    if print_cost and i% 1000 = 0:
        print("Cost after iteration %i: %f" %(i,cost))
        
    return parameters

def predict(parameters,X):
    A2,cache = forward_propgation(X,parameters)
    prediction = (A2 > 0.5)
    
    return prediction

parameters = nn_model(X,Y,num_iteration = 10000,learing_rate = 1.2,print_cost = False)
