import numpy as np
import matplotlib.pyplot as plt

def init_w(training_set):
    return np.zeros(training_set.shape[1])

def loss(x,weights,y):
    return np.sum((x.dot(weights)-y)**2)/2

def gradLoss(x,weights,y):
    gradient=[]
    for i in xrange(len(weights)):
        gradient.append(np.sum(y*x[:,i]-x.dot(weights)*x[:,i]))
    return np.array(gradient)

def add_offset(x):
    n,d=x.shape
    return np.hstack((np.ones((n,1)),x))



def update_weights(weights,x,y,alpha,convergence_rate=0.001):
    total_loss = loss(x,weights,y)
    print "total_slo: "+str(total_loss)
    while total_loss>convergence_rate:
        print "weights: " + str(weights)
        weights = weights-alpha*gradLoss(x,weights,y)

        total_loss = loss(x, weights, y)
    return weights

def optimal_weights(x,y):
    feat_matrix=x.T.dot(x)
    sum_y_x = np.sum(y * x,0)
    #print sum_y_x
    weights=np.linalg.inv(feat_matrix).dot(sum_y_x)
    return weights.reshape(weights.shape[0],1)

def get_predictions(x,weights):
    return x.dot(weights)

def prediction_loss(y_pred,y):
    return np.sum((y_pred-y)**2)/2


if __name__=="__main__":
    #F = open("test_data.txt", "r")
    f= np.loadtxt("test_data.txt",usecols=([1,2,3,4]))

    # Separating in training and test set:
    x=f[:,[0,1,2]]
    y = f[:, 3]
    y = y.reshape(y.shape[0],1)
    # Adding column for the offset:
    x=add_offset(x)

    weights = optimal_weights(x,y)
    y_pred = get_predictions(x,weights)

    print "loss: "+str(prediction_loss(y_pred,y))




    # alpha=2
    # update_weights(weights,x,y,alpha)
    # print gradLoss(x,weights,y)

