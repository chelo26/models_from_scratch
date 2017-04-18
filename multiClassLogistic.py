import numpy as np
import matplotlib.pyplot as plt
import os
from __future__ import division
from sklearn.datasets import load_iris
from sklearn import preprocessing


# Scaling the function
def scaling(x):
    x=x-np.mean(x,0)
    x=x/np.std(x,0)
    return x

# Adding Offset:
def add_offset(x):
    n,d=x.shape
    return np.hstack((np.ones((n,1)),x))

# Initialization:
def init_weights(x,y):
    dim = x.shape[1]
    num_class=y.shape[1]
    return np.random.randn(dim, num_class)/dim

# Loss:

# correct solution:
def softmax(x,weights):
    a=x.dot(weights)
    e_a = np.exp(a)
    sum_e_a = np.sum(e_a,1).reshape(x.shape[0],1)
    return e_a/sum_e_a# only difference

def crossEntropyLoss(weights,x,y):
    return -np.sum(y*np.log(softmax(x,weights)))

def optimize_weights(x,y,alpha=0.0003,num_it=5000):
    weights=init_weights(x,y)
    for j in xrange(num_it):
        for i in xrange(weights.shape[1]):
            weights[:,i] -= -alpha*gradientLoss(y,x,weights,i)
        print "loss: "+str(crossEntropyLoss(weights,x,y))
        print "weights: "+str(weights)
    return weights

def gradientLoss(t,x,weights,i):
    y=softmax(x,weights)
    dif=t-y
    dif_i=dif[:,i].reshape(dif.shape[0],1)
    grad=dif_i*x
    return np.sum(grad,0)


if __name__=="__main__":

    # Using iris dataset:

    data = load_iris()
    x=data.data
    x=add_offset(x)
    y=data.target

    # Encoding:
    lb = preprocessing.LabelBinarizer()
    lb.fit(y)
    t=lb.transform(y)

    # Variables:
    n=x.shape[0]
    dim=x.shape[1]

    # Initialize weights:
    weights=optimize_weights(x,t)














