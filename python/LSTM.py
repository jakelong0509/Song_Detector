import numpy as np
import os
import sys
from functions import activations as act, helper_func as func
from wrapper import Bidirectional

class LSTM():
    def __init__(self, input_dim, output_dim):
        """
        input_dim: dimension of input data (Tx, n_x)
        output_dim: dimension of output hidden state (Tx, n_a)
        """

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.params = {}

        # a toggle used to decide LSTM_backward
        self.is_backward = False

        # retrieve dimension
        _, _, self.n_x = self.input_dim
        _, _, self.n_a = self.output_dim

        # initialize cell params
        _re_cell_W = func.orthogonal(self.n_a)
        _ke_cell_W = func.xavier((self.n_a, self.n_x)) # W.shape = (n_a, n_x)
        _cell_b = np.zeros((1,self.n_a))
        _cell_W = np.concatenate((_re_cell_W, _ke_cell_W), axis = 1) # shape = (n_a, na + n_x)
        self.params["Wc"] = _cell_W
        self.params["bc"] = _cell_b

        # initialize update params
        _re_u_W = func.orthogonal(self.n_a)
        _ke_u_W = func.xavier((self.n_a, self.n_x))
        _u_b = np.zeros((1,self.n_a))
        _u_W = np.concatenate((_re_u_W, _ke_u_W), axis = 1) # shape = (n_a, na + n_x)
        self.params["Wu"] = _u_W
        self.params["bu"] = _u_b

        # initialize forget params
        _re_f_W = func.orthogonal(self.n_a)
        _ke_f_W = func.xavier((self.n_a, self.n_x))
        _f_b = np.zeros((1,self.n_a))
        _f_W = np.concatenate((_re_f_W, _ke_f_W), axis = 1) # shape = (n_a, n_a + n_x)
        self.params["Wf"] = _f_W
        self.params["bf"] = _f_b

        # initialize output params
        _re_o_W = func.orthogonal(self.n_a)
        _ke_o_W = func.xavier((self.n_a, self.n_x))
        _o_b = np.zeros((1, self.n_a))
        _o_W = np.concatenate((_re_o_W, _ke_o_W), axis = 1) # shape = (n_a, n_a + n_x)
        self.params["Wo"] = _o_W
        self.params["bo"] = _o_b

    def cell_forward(self, a_prev, c_prev, xt):
        """
        a_prev: hidden state of previous iteration (1, n_a)
        c_prev: cell memory of previous iteration (1, n_a)
        xt: current data (1, n_x)
        """
        concat = np.concatenate((a_prev, xt), axis = 1)
        ctt = act.tanh(np.matmul(concat, np.transpose(self.params["Wc"])) + self.params["bc"])
        fu = act.sigmoid(np.matmul(concat, np.transpose(self.params["Wu"])) + self.params["bu"])
        ff = act.sigmoid(np.matmul(concat, np.transpose(self.params["Wf"])) + self.params["bf"])
        fo = act.sigmoid(np.matmul(concat, np.transpose(self.params["Wo"])) + self.params["bo"])
        ct = np.multiply(fu, ctt) + np.multiply(ff, c_prev)
        at = np.multiply(fo, act.tanh(ct))

        cache = (ctt, fu, ff, fo, ct, at, a_prev, c_prev)
        return at, ct, cache

    def forward_propagation(self, X):
        """
        X: data of 1 row (Tx, n_x)
        -----------------------------
        backward case:
            a_prev = a_next
            c_prev = c_next
        """
        Tx, n_x = X.shape
        caches = []
        self.A = []
        a0 = np.zeros((1, self.n_a))
        c0 = np.zeros((1, self.n_a))

        a_prev = a0
        c_prev = c0

        range = []
        if self.is_backward:
            range = np.flip(np.arange(Tx), axis = 0)
        else:
            range = np.arange(Tx)

        for t in range:
            xt = np.atleast_2d(X[t, :])
            at, ct, cache = self.cell_forward(a_prev, c_prev, xt)
            a_prev = at

            c_prev = ct
            self.A.append(at.reshape(-1))
            caches.append(cache)

        return np.array(self.A), caches
if __name__ == "__main__":
    input_dim = (10,10,3)
    output_dim = (10,10,5)

    X = np.random.random((10,10,3))
    bi_LSTM = Bidirectional(LSTM(input_dim, output_dim), X[1,:,:])
    A = bi_LSTM.concatLSTM()
    print(A.shape)
