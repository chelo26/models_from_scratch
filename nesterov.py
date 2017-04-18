from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import load_iris
from sklearn import preprocessing
#from tensorflow.examples.tutorials.mnist import input_data


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

def softmax_derivative(x,weights):
    return softmax(x,weights)*(1-softmax(x,weights))

def crossEntropyLoss(weights,x,y):
    return -np.sum(y*np.log(softmax(x,weights)))

def optimize_weights(x,y,alpha=0.001,num_it=5000):
    weights=init_weights(x,y)
    for j in xrange(num_it):
        for i in xrange(weights.shape[1]):
            weights[:,i] -= -alpha*gradientLoss(y,x,weights,i)
        print "loss: "+str(crossEntropyLoss(weights,x,y))
        #print "weights: "+str(weights)
    return weights

def gradientLoss(t,x,weights,i):
    y=softmax(x,weights)
    dif=(t-y)
    dif_i=dif[:,i].reshape(dif.shape[0],1)
    soft_der=softmax_derivative(x, weights)
    soft_der=soft_der[:, i].reshape(soft_der.shape[0],1)
    grad=dif_i*soft_der*x
    return np.sum(grad,0)

# Making predictions and testing:

def predict(weights,x,y):
    return np.argmax(softmax(x,weights),1)

def get_errors(t,y):
    return np.sum(t==y)/t.shape[0]

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

    # Predictions:
    y_hat=predict(weights,x,t)

    error=get_errors(y,y_hat)

    print error


    #
    # print "start:"
    # data = input_data.read_data_sets("../data/MNIST/", one_hot=True)
    # print "data loaded"
    # training_features = data.train.images
    # training_labels = data.train.labels
    #
    # # Creating a list of features and labels:
    #
    # print training_features.shape
    # print training_labels.shape
    #
    # training_features = add_offset(training_features)
    # weights = optimize_weights(training_features,training_labels)
    #
    #







