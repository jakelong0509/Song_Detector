import numpy as np

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def tanh(Z):
    return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
