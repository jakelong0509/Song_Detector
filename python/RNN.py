import numpy as np
import math
import copy
import wrapper
from data_preprocessing import song_preprocessing as sp
from numpy import linalg as LA
from functions import activations as act, helper_func as func

Tx = sp.get_Tx("../songs/")
print(Tx)

class RNN():
    def __init__(self, input_dim, output_dim, kernel_initializer = "Xavier", recurrent_initializer = "ortho", return_sequences = False):
        """
            params
            -------------------------
            input_dim: dimension of input data (None ,Tx, n_x) [n_x = 101]
            output_dim: dimension of output data (None, Tx, n_a)
            initializer: Xavier initializer
            return_sequences: return the last output of output sequence or return output at every node
            last_layer: if the layer is the last layer in model in order to initialize Wy
        """
        _, _, n_a = output_dim
        _, _, n_x = input_dim


        # initialize W and ba
        if kernel_initializer == None:
            self.Wx = np.random.random((n_a, n_x))

        elif kernel_initializer == "Xavier":
            standard_dev = np.sqrt(6/(n_x + n_a))
            self.Wx = np.random.uniform(standard_dev, -standard_dev, (n_a, n_x))

        if recurrent_initializer == None:
            self.Wa = np.random.random((n_a, n_a))
        elif recurrent_initializer == "ortho":
            self.Wa = func.orthogonal(n_a)
        self.ba = np.zeros((1,n_a))
        self.W = np.concatenate((self.Wa, self.Wx), axis = 1) #(n_a, n_a + n_x)



    def cell_forward(self, a_prev, xt):
        """
        a_prev: previous hidden state (1, n_a)
        xt: current input (1, n_x)
        """
        X = np.concatenate((a_prev, xt), axis = 1) # (1, n_a + n_x)
        curr_z = np.matmul(X, np.transpose(self.W)) + self.ba
        a_next = act.tanh(curr_z)

        cache = (curr_z, a_next)
        return a_next, cache

    def layer_forward(self, a0, X, is_backward = False):
        """
        a0: initial hidden state (1,n_a)
        X: input data (Tx, n_x)
        """
        Tx, n_x = X.shape
        range = []
        if is_backward:
            range = np.flip(np.arange(Tx), axis = 0)
        else:
            range = np.arange(Tx)


        a_prev = a0 # in backward a0 = a_Tx+1
        caches = []
        for t in range:
            xt = np.atleast_2d(X[t,:])
            a_next, cache = self.cell_forward(a_prev, xt)
            a_prev = a_next
            caches.append(cache)

        return caches


if __name__ == "__main__":
    input_dim = (10, 10, 3)
    output_dim = (10,10,5)
    a0 = np.zeros((1,5))
    X = np.random.random((10,3))
    bi_LSTM = wrapper.Bidirectional(LSTM(input_dim, output_dim), a0, X)


    A = bi_LSTM.concatLSTM()
    print(A)
