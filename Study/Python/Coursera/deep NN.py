# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:19:01 2017

@author: Surface
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases_v3 import *
from dnn_utils_v2 import sigmoid ,sigmoid_backward,relu ,relu_backward

def initialize_parameters(n_0,n_1,n_2):
    
    np.random.seed(1)
    W1 = np.random.randn((n_0,n_1)) * 0.01
    b1 = np.random.zeros((n_1,1))
    W2 = np.random.randn((n_2,n_1)) * 0.01
    b2 = np.random.zeros((n_2,1))
    
    parameters = {"W1" : W1,
                  "b1" : b1,
                  "W2" : W2,
                  "b2" : b2}
    
    return parameters

#前一种函数的推广，可以为多层
def initialize_parameters_deep(layer_dims):
    
    #layer_dims为多维数组，维度表示层数，数表示节点
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    
    for i in range(1,L):
        parameters['W' + str(l)] = np.random.randn((layer_dims[l],layer_dims[l-1])) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
        
    return parameters

def linear_forward(A,W,b):
    
    Z = np.dot(W,A) + b
    
    cache = (A,W,b)
    
    return Z,cache

def linear_activation_forward(A_prev,W,b,activation):
    
    if activation == "sigmoid":
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = sigmoid(Z)
        #sigoid 和 relu的返回值是计算出来的 A 和 原始数据 Z
    elif activation =="relu":
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = relu(Z)
        
    cache = (linear_cache,activation_cache)
    
    return A,cache

def L_model_forward(X,parameters):
    
    caches = []
    A = X
    L = len(parameters) // 2  #整数除法 对字典使用len()表示计算字典元素的个数
    
    for i in  range(1,L):
        A_prev = A
        A,cache = linear_activation_forward(A_prev,parameters['W' + str(l)], parameters['b' + str(l)], activation = 'relu')
        caches.append(cache)
        
    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], activation = 'sigmoid')
    caches.append(cache)
    
    return AL, caches

def compute_cost(AL,Y):
    
    m = Y.shape[1]
    
    logprobs = np.multiply(Y,np.log(AL)) + np.multiply((1 - Y),np.log(1 - AL))
    cost = - np.sum(logprobs) / m
    
    cost = np.squeeze(cost) #保险，个人觉得其实没必要
    
    return cost

def linear_backward(dZ,cache):
    
    A_prev,W,b = cache
    m = A_prev.shape[1]
    
    dW = np.dot(dZ,A_prev.T) / m
    db = np.sum(dZ,axis = 1,keepdims = True) / m
    dA_prev = np.dot(W.T,dZ)
    
    return dA_prev,dW,db

def linear_activation_backward(dA,cache,activation):
    
    linear_cache,activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev,dW,db = linear_backward(dZ,linear_cache)
        
    return dA_prev,dW,db

def L_model_backward(AL,Y,caches):
    
    grads ={}
    L = len(caches) #层数
    m = AL.shape[1]
    
    grads["dA" + str(L)] = -(np.divide(Y,AL) - np.divide(1 - Y,1 - AL)
    #last unit dAL
    current_cache = caches[L-1]
    # 等于第L层的参数，计算第 L - 1层的dA，
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(grads["dA" + str(L)], current_cache, activation = 'sigmoid')

    for l in reversed(range(L-1)):  #first i = L - 2
        current_cache = caches[l] # 第L - 1层的参数 第一次循环中 l+1 = L-1
        grads["dA" + str(l)],grads["dW" + str(l + 1)],grads["db" + str(l + 1)] = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = 'relu')
    
    return grads

def update_parameters(parameters,grads,learning_rate):
    
    L = len(parameters) // 2
    
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    
    return parameters
