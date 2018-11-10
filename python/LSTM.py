import numpy as np
import RNN
import os
import sys
from activation import activations as act
class LSTM(RNN):
    def __init__(self, input_dim, output_dim):
        RNN.__init__(self, input_dim, output_dim)

    def layer_forward_for(self, a0, X):
        """
        X: dataset (Tx, n_x)
        """
        return self.layer_forward(a0, X)

if __name__ == "__main__":
    input_dim = (10, 10, 101)
    output_dim = (10,10,128)
    a0 = np.zeros((1,128))
    X = np.random.random((10,101))
    LSTM = LSTM(input_dim, output_dim)
    caches = LSTM.layer_forward_for(a0,X)
    for cache in caches:
        _, a = cache
        print(a)
