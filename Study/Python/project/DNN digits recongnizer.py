# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:58:04 2017

@author: Surface
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:43:19 2017

@author: Surface
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 20:25:06 2017

@author: Surface
"""

import numpy as np
import matplotlib.pyplot as plt
import math


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)-1#numberOfLines为样本数
    returnMat = np.zeros((numberOfLines,28*28))#创建一个（样本数,特征值）的数组
    classLabel = np.zeros((numberOfLines,1))#label
    index = 0
    for line in arrayOLines:
        if index == 0:
            index += 1
            continue
        line = line.strip()#去掉回车
        listFromLine = line.split(',')#以,分割
        classLabel[index-1,:] = listFromLine[0]#第一个为label,为整数
        #print(len(listFromLine))
        returnMat[index-1,:] = listFromLine[1:]#将剩下的传递给dataset数组
        
        index += 1
    
    return returnMat,classLabel


def one_hot(Labels):
    one_hot = np.zeros((10,Labels.shape[1]))
    for i in range(Labels.shape[1]):
        k = Labels[:,i]
        one_hot[int(k),i] = 1

    return one_hot

def random_mini_batches(X,Y,mini_batch_size = 64):
    
    m = X.shape[1]
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:,permutation]
    shuffled_Y = Y[:,permutation]
    
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0,num_complete_minibatches):
        mini_batch_X = shuffled_X[:,mini_batch_size * k : mini_batch_size * (k+1)]
        mini_batch_Y = shuffled_Y[:,mini_batch_size * k : mini_batch_size * (k+1)]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:,-(m - mini_batch_size * math.floor(m / mini_batch_size)):]
        mini_batch_Y = shuffled_Y[:,-(m - mini_batch_size * math.floor(m / mini_batch_size)):]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        
    return mini_batches

def initialization(layer_dims):
    
    parameters = {}
    L = len(layer_dims)
    for l in range(1,L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
        
    return parameters

def initialize_adam(parameters):
    
    L = len(parameters) // 2
    v = {}
    s = {}
    
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b" + str(l + 1)].shape)
        s["dW" + str(l+1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters["b" + str(l + 1)].shape)
    
    return v,s


def sigmoid(z):
    
    s = 1 / (1 + np.exp(-z))
    return s,z

def relu(z):
    
    s = np.maximum(0,z)
    
    return s,z

def softmax(Z):
    
    t = np.exp(Z)
    A = t / np.sum(t,axis = 0,keepdims = True)
    
    return A,Z

def linear_forward(A,W,b):
    
    Z = np.dot(W,A) + b
    
    cache = (A,W,b)
    
    return Z,cache

def linear_activation_forward(A_prev,W,b,activation,keep_prob = 0.8):
    
    if activation == "sigmoid":
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = sigmoid(Z)
        D = np.random.rand(A.shape[0],A.shape[1])
        D = (D < keep_prob)
        A = A * D
        A = A / keep_prob
        #sigoid 和 relu的返回值是计算出来的 A 和 原始数据 Z
    elif activation =="relu":
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = relu(Z)
        D = np.random.rand(A.shape[0],A.shape[1])
        D = (D < keep_prob)
        A = A * D
        A = A / keep_prob
      
    elif activation =="softmax":
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = softmax(Z)
        D = np.random.rand(A.shape[0],A.shape[1])
        D = (D < 1)
        
    cache = (linear_cache,activation_cache,D)
    
    return A,cache

def L_model_forward(X,parameters,keep_prob = 0.8):
    
    caches = []
    A = X
    L = len(parameters) // 2  #整数除法 对字典使用len()表示计算字典元素的个数
    
    for l in  range(1,L):
        A_prev = A
        A,cache = linear_activation_forward(A_prev,parameters['W' + str(l)],parameters['b' + str(l)], activation = 'relu',keep_prob = 0.8)
        caches.append(cache)
        
        
    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], activation = 'softmax',keep_prob = 0.8)
    caches.append(cache)
    
    return AL, caches

def compute_cost_with_regularization(AL,Y,parameters,lambd):
    
    m = Y.shape[1]
    summ = 0
    
    logprobs = np.multiply(Y,np.log(AL))
    cross_entropy_cost = - np.sum(logprobs) / m 
    
    cross_entropy_cost = np.squeeze(cross_entropy_cost) 
    
    for l in range(1,(len(parameters) // 2) + 1):
        summ += np.sum(np.square(parameters["W" + str(l)]))
    
    L2_regularization_cost = lambd * summ / (2 + m)
    
    cost = cross_entropy_cost + L2_regularization_cost
    
    
    return cost

def relu_backward(dA,cache):
    
    Z = cache
    dZ = np.array(dA, copy=True)
    
    dZ[Z <= 0] = 0
    
    return dZ

def sigmoid_backward(dA,cache):
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    return dZ

def softmax_backward(dA,cache):
    
    Z = cache
    A,Z = softmax(Z)
    dAl = A
    dAindex = np.argwhere(dA != 0)
    dAindex = dAindex[dAindex[:,1].argsort()]
    for i in range(dA.shape[1]):
        j = dAindex[i][0]
        dAl[:,i][j] = dAl[:,i][j] - 1
    
    return dAl



def linear_backward(dZ,cache,lambd):
    
    A_prev,W,b= cache
    m = A_prev.shape[1]
    
    dW = np.dot(dZ,A_prev.T) / m +  + lambd * W / m
    db = np.sum(dZ,axis = 1,keepdims = True) / m
    dA_prev = np.dot(W.T,dZ)
    return dA_prev,dW,db

def linear_activation_backward(dA,cache,lambd,activation,keep_prob = 0.8):
    
    linear_cache,activation_cache,D = cache
    
    if activation == "relu":
        dA = dA * D
        dA = dA / keep_prob
        dZ = relu_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache,lambd)

    elif activation == "softmax":
        dZ = softmax_backward(dA,activation_cache)
        dA_prev,dW,db = linear_backward(dZ,linear_cache,lambd)
        
        
    return dA_prev,dW,db


def L_model_backward(AL,Y,caches,lambd,keep_prob = 0.8):
    
    grads = {}
    L = len(caches)
    grads["dA" + str(L)] = -np.divide(Y,AL)
    current_cache = caches[L-1]
    
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(grads["dA" + str(L)], current_cache,lambd,activation = 'softmax',keep_prob = 0.8)
    
    for l in reversed(range(L-1)):  #first i = L - 2
        current_cache = caches[l] # 第L - 1层的参数 第一次循环中 l+1 = L-1
        grads["dA" + str(l)],grads["dW" + str(l + 1)],grads["db" + str(l + 1)] = linear_activation_backward(grads["dA" + str(l + 1)], current_cache,lambd,activation = 'relu',keep_prob = 0.8)
        
    return grads


def update_parameters(parameters,grads,learning_rate):
    
    L = len(parameters) // 2
    
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    
    return parameters


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate,beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    
    L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    for l in range(L):
        
        v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1 - beta1) * grads['dW' + str(l+1)]
        v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1 - beta1) * grads['db' + str(l+1)]

        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1 - beta1 ** t)
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1 - beta1 ** t)
    
        s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1 - beta2) * grads['dW' + str(l+1)] * grads['dW' + str(l+1)]
        s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1 - beta2) * grads['db' + str(l+1)] * grads['db' + str(l+1)]
        
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1 - beta2 ** t)
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1 - beta2 ** t)
        
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_corrected["dW" + str(l+1)] / (np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon)
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_corrected["db" + str(l+1)] / (np.sqrt(s_corrected["db" + str(l+1)]) + epsilon)
       
    return parameters, v, s

def L_layer_model(X,Y,layer_dims,learning_rate,lambd,num_epochs,mini_batch_size = 128,keep_prob = 0.8,print_cost = True):
    
    costs = []
    parameters = initialization(layer_dims)
    t = 0
    v, s = initialize_adam(parameters)
    for i in range(0,num_epochs):
        
        #minibatches = random_mini_batches(X,Y,mini_batch_size)
        
        #for minibatch in minibatches:
        #    (minibatch_X, minibatch_Y) = minibatch
            
        #    AL,caches = L_model_forward(minibatch_X,parameters,keep_prob = 0.8)

        #    cost = compute_cost_with_regularization(AL,minibatch_Y,parameters,lambd)
        
         #   grads = L_model_backward(AL,minibatch_Y,caches,lambd,keep_prob = 0.8)
        
            #t = t + 1 # Adam counter
            #parameters, v, s = update_parameters_with_adam(parameters, grads, v,s,t, learning_rate)
         #   parameters = update_parameters(parameters,grads,learning_rate)
            
        AL,caches = L_model_forward(X,parameters,keep_prob = 0.8)

        cost = compute_cost_with_regularization(AL,Y,parameters,lambd)
        
        grads = L_model_backward(AL,Y,caches,lambd,keep_prob = 0.8)
        
        t = t + 1 # Adam counter
        parameters, v, s = update_parameters_with_adam(parameters, grads, v,s,t, learning_rate)
        #parameters = update_parameters(parameters,grads,learning_rate)
        if print_cost and i % 100 == 0:
            print ("Cost after epoch %i: %f" %(i, cost))
            costs.append(cost)
    
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('epochs (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
    
def predict(X,Y, parameters):
   
    m = X.shape[1] 
    AL, caches = L_model_forward(X, parameters)
    
    for i in range(AL.shape[1]):
        k = np.max(AL[:,i])
        for j in range(AL.shape[0]):
            if AL[j,i] == k:
                AL[j,i] = 1
            else:
                AL[j,i] = 0
    count0 = AL - Y

    wrong = np.sum(count0 != 0) / 2
                
    print("Accuracy: "  + str(1 - wrong/m))

returnMat,classLabel = file2matrix("train.csv")

returnMat = returnMat.reshape(returnMat.shape[0],-1).T
classLabel = classLabel.reshape(classLabel.shape[0],-1).T

Mat = returnMat
Label = one_hot(classLabel)

testMat = Mat[:,:-2000]
testLabel = Label[:,:-2000]
Mat = Mat[:,:40000]
Label = Label[:,:40000]
layer_dims = [Mat.shape[0],100,50,20,10]
parameters = L_layer_model(Mat,Label,layer_dims,learning_rate = 0.001,lambd = 0.3,num_epochs = 2000,mini_batch_size = 128,keep_prob = 0.6, print_cost = True)
print("训练样本：")
prediction = predict(Mat,Label,parameters)
print("测试样本：")
prediction = predict(testMat,testLabel,parameters) 