import numpy as np
from numpy import linalg as LA

def orthogonal(shape):
    A = np.random.rand(shape,shape)
    P = (A + np.transpose(A))/2 + shape*np.eye(shape)

    vals, vecs = LA.eig(P)
    w = vecs[:,0:shape]

    return w
