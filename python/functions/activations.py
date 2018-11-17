import numpy as np

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def hard_sigmoid(Z):
    return np.atleast_2d(max(0, min(1, x*0.2 + 0.5)))

def tanh(Z):
    return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

def relu(Z):
    return np.atleast_2d(max(0,Z))

def softmax(Z):
    deli = np.sum(np.exp(Z))
    return np.exp(Z) / deli
