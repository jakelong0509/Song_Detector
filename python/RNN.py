import numpy as np
import math
from data_preprocessing import song_preprocessing as sp
from numpy import linalg as LA
from activation import activations as act

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
            self.Wa = orthogonal(n_a)
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

    def layer_forward(self, a0, X):
        """
        a0: initial hidden state (1,n_a)
        X: input data (Tx, n_x)
        """
        Tx, n_x = X.shape
        a_prev = a0
        caches = []
        for t in range(Tx):
            xt = np.atleast_2d(X[t,:])
            print(xt.shape)
            a_next, cache = self.cell_forward(a_prev, xt)
            a_prev = a_next
            caches.append(cache)

        return caches

# Create LSTM layer inherit from standard RNN layer
class LSTM(RNN):
    def __init__(self, input_dim, output_dim, is_backward = False):
        RNN.__init__(self, input_dim, output_dim)
        self.is_backward = is_backward

    def layer_forward_for(self, a0, X):
        """
        X: dataset (Tx, n_x)
        """
        if self.is_backward:
            return self.layer_forward(a0, X)
        else:
            """
            in backward LSTM a0 = a_Tx+1
            """
            Tx, n_x = X.shape
            a_next = a0
            count = Tx - 1
            caches_back = []
            while(count > 0):
                xt = np.atleast_2d(X[count, :])
                a_prev, cache = self.cell_forward(a_next, xt)
                a_next = a_prev
                caches_back.append(cache)
                count -= 1

            return caches_back


    # concat forward hidden state and backward hidden state
    def concatLSTM(self):
        self.A = []

        for c, cb in self.caches_forward, self.caches_back:
            _, a_forward = c
            _, a_backward = cb
            concat = np.concatenate((a_forward, a_backward), axis = 1)
            self.A.append(concat)
        return self.A

def orthogonal(shape):
    A = np.random.rand(shape,shape)
    P = (A + np.transpose(A))/2 + shape*np.eye(shape)

    vals, vecs = LA.eig(P)
    w = vecs[:,0:shape]

    return w


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
