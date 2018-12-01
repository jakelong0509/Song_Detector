import numpy as np
import sys
import os
from numpy import linalg as LA

# Orthogonal initializer
def orthogonal(shape):
    A = np.random.rand(shape,shape)
    P = (A + np.transpose(A))/2 + shape*np.eye(shape)

    vals, vecs = LA.eig(P)
    w = vecs[:,0:shape]

    return w

# Xavier initializer
def xavier(dimension):
    # checking if dimension is a tuple or not
    if not isinstance(dimension, tuple):
        sys.exit("Argument 'dimension' is not a tuple. Terminating....")

    n_out, n_in = dimension
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, size = dimension)

# duplicate hidden state
def duplicate(S, n_s, _prev_s, axis):
    # Parameter:
    # S: width of windows
    # n_s: dimension of hidden state of post_LSTM
    _prev_S = None
    if axis == 0:
        ones = np.ones((S, n_s))
        _prev_S = ones * _prev_s
    elif axis == 1:
        ones = np.ones((n_s, S))
        _prev_S = ones * _prev_s
    return _prev_S

def t_lost(y_true, y_hat):
    """
    y_true: true label of dataset (n_y,) [one hot]
    y_hat: predicted value at time step t (n_y,)
    """
    return -(np.sum(y_true * np.log(y_hat)))

#def adam_optimizer(lr, beta1, beta2, decay):


def normalization(X, Tx):
    mean = 1/Tx* np.sum(X, axis = 0)
    X = X - mean
    variance = 1/Tx * np.sum(X**2, axis = 0)
    X = X/variance
    return X

def dictionary_to_vector(dict):
    vector = []
    keys_shape = {}
    for k, v in dict.items():
        keys_shape[k] = v.shape
        for r in dict[k]:
            for i in r:
                vector.append(i)

    return vector, keys_shape

def vector_to_dictionary(vector, keys_shape):
    dict = {}
    for k,v in keys_shape.items():
        row, col = v
        vec_num = row * col
        vec_temp = vector[:vec_num]
        remain = vector[vec_num:]
        dict[k] = vec_temp.reshape(v)
        vector = remain
    
    return vector, dict
