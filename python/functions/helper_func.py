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
