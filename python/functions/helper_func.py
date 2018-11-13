import numpy as np
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
