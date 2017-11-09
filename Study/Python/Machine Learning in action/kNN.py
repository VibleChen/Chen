# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 21:15:29 2017

@author: Surface
"""

import numpy as np
import operator
import os

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0.0,0.0],[0.0,0.1]])
    labels = ['C','A','B','D']
    return group,labels

"""
关于read()方法：
1、读取整个文件，将文件内容放到一个字符串变量中
2、如果文件大于可用内存，不可能使用这种处理

关于readline()方法：
1、readline()每次读取一行，比readlines()慢得多
2、readline()返回的是一个字符串对象，保存当前行的内容

关于readlines()方法：
1、一次性读取整个文件。
2、自动将文件内容分析成一个行的列表。
"""

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)#numberOfLines为样本数
    returnMat = np.zeros((numberOfLines,3))#创建一个（样本数,特征值）的数组
    classLabelVector = []#label列表
    index = 0
    for line in arrayOLines:
        line = line.strip()#去掉回车
        listFromLine = line.split('\t')#以/t分割
        #print(type(listFromLine[0]))
        returnMat[index,:] = listFromLine[0:3]#将前三个传递给dataset数组
        #returnMat = listFromLine[0:3]此处与上处不同点在于，上处传递给某一维度，所以为数字，若不带角标，默认为传递字符串
        #print(type(returnMat[index][0]))
        classLabelVector.append(int(listFromLine[-1]))#最后一个为label,为整数
        index +=1
    
    return returnMat,classLabelVector

def normalization(dataset):
    minvals = dataset.min(0)
    maxvals = dataset.max(0)
    ranges = maxvals - minvals
    normdataset = np.zeros(np.shape(dataset))
    m = dataset.shape[0]
    normdataset = dataset - np.tile(minvals,(m,1))
    normdataset = normdataset / np.tile(ranges,(m,1))
    return normdataset,ranges,minvals


def classify(X,dataSet,labels,k):
    #数据的格式为(样本数，特征值)
    m = dataSet.shape[0]
    diffMat = np.tile(X,(m,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)#列相加
    distances = sqDistances ** 0.5
    #距离计算完成
    sortedDisIndicies = distances.argsort()#索引排序
    
    classCount = {}#创建一个空字典，key为label，值为出现的次数
    for i in range(k):
        voteLabel = labels[sortedDisIndicies[i]]#voteLabel是按距离排序的字符
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1#.get(key,default)按照key寻找value，若不存在则设为默认值
    #现在为止，我们已经将排好序的dataset和前k个label对应起来
    #classCount的格式为{距离最近的key1:出现的次数,第二近的key2 : 出现的次数...}
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
    #sorted(iterable,cmp,key,reverse)
    #cmp为用于比较的函数
    #key = operator.itemgetter()函数用于获取对象的哪些维的数据，key相当于排序的维度，默认为1
    #print(classCount)
    #print(sortedClassCount)
    
    return sortedClassCount[0][0]

def datingClassTest():
    
    horatio = 0.1
    dataset,labels = file2matrix('datingTestSet3.txt')
    normMat,ranges,minvals = normalization(dataset)
    m = dataset.shape[0]
    numTestVecs = int(m * horatio)#选取了样本的前10%进行预测，后90%作为样本集
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify(normMat[i,],normMat[numTestVecs:m,:],labels[numTestVecs:m],3)
        print("the classifier came back with :%d, the real answer is : %d" %(classifierResult,labels[i]))
    
    if (classifierResult != labels[i]):
        errorCount += 1
    print("the total error rate is : %f" %(errorCount / numTestVecs))

def classifyperson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(input("percentage of time spent playing video game:"))
    ffMiles = float(input("frequent filer miles earned per year:"))
    iceCream = float(input("ilters of ice cream consumed per year:"))
    dataMat,labels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minvals = normalization(dataMat)
    inArr = np.array([ffMiles,percentTats,iceCream])
    classifierResult = classify(((inArr - minvals) / ranges),dataMat,labels,3)
    print("you will probably like this person:",resultList[classifierResult - 1])
    
def img_to_vector(filename):# 32*32
    returnVector = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVector[0,32*i+j] = int(lineStr[j])
    #print(type(lineStr))
    return returnVector

def handwritingClassTest():
    hwlabels = []
    trainingFileList = os.listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNumStr = int(fileNameStr.split('_')[0])#fileName为label_样本
        hwlabels.append(classNumStr)
        trainingMat[i] = img_to_vector('trainingDigits/%s' %fileNameStr)
    
    testFileList =os.listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumStr = int(fileNameStr.split('_')[0])
        vectorUnderTest = img_to_vector('testDigits/%s' %fileNameStr)#vectorUnderTest的shape为(1,1024)
        classifierResult = classify(vectorUnderTest,trainingMat,hwlabels,3)
        print("the classifier came back with:%d,the real answer is :%d" %(classifierResult,classNumStr))
        if (classifierResult !=classNumStr):
            errorCount += 1
    
    print("\nthe total number of error is %d" %errorCount)
    print("\nthe total error rate is :%f" %(errorCount / float(mTest)))
        