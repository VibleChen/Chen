# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 21:17:48 2017

@author: Surface
"""

import kNN
import matplotlib.pyplot as plt

group,labels = kNN.createDataSet()
#print(group,labels)
#print(classify([0,0],group,labels,2))

dataset,labels = kNN.file2matrix("datingTestSet2.txt")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataset[:,1],dataset[:,2],15*np.array(labels),15*np.array(labels))
plt.show()

#kNN.datingClassTest()
#kNN.classifyperson()
kNN.handwritingClassTest()