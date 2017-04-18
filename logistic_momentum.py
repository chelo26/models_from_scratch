from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import load_iris
from sklearn import preprocessing
#from tensorflow.examples.tutorials.mnist import input_data
from sklearn.datasets import make_classification


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
def sigmoid(x,weights):
    a=x.dot(weights)
    a = np.exp(-a)
    return 1/(1+a)# only difference

def sigmoid_derivative(x,weights):
    return sigmoid(x,weights)*(1-sigmoid(x,weights))

def crossEntropyLoss(weights,x,y):
    return -np.sum(y*np.log(sigmoid(x,weights)))

def optimize_weights(x,y,alpha=0.00001,num_it=100,mu=0.85):
    weights=init_weights(x,y)
    k=0
    old_grad= gradientLoss(y, x, weights)
    for j in xrange(num_it):
        new_grad = mu*old_grad + gradientLoss(y,x,weights)
        #new_grad = gradientLoss(y, x, weights)
        weights -= alpha*new_grad
        if k%10==0:
            print "loss: "+str(crossEntropyLoss(weights,x,y))
        k+=1
        old_grad=new_grad
        #print "weights: "+str(weights)
    return weights

def gradientLoss(y_true,x,weights):
    y_pred=sigmoid(x,weights)
    dif=y_true-y_pred
    grad = dif*sigmoid_derivative(x, weights)*x
    sum_grad = -np.sum(grad, 0).reshape(-1,1)
    return sum_grad

# Making predictions and testing:

def predict(weights,x,y,threshold=0.5):
    results=sigmoid(x,weights)>threshold
    return results.astype(int)

def get_accuracy(t,y):
    return np.sum(t==y)/t.shape[0]

if __name__=="__main__":

    # Using iris dataset:

    x_gen,y=make_classification(n_samples=5000,n_features=5,n_classes=2)
    #plt.scatter(x_gen[:, 0], x_gen[:, 1], marker='o', c=y)

    x=add_offset(x_gen)

    y=y.reshape(-1,1)

    # Variables:
    n=x.shape[0]
    dim=x.shape[1]

    # Initialize weights:
    weights=optimize_weights(x,y)

    # Predictions:
    y_hat=predict(weights,x,y)

    accuracy=get_accuracy(y,y_hat)

    print accuracy


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







