import numpy as np
import os
import sys
from functions import activations as act, helper_func as func
from wrapper import Bidirectional
import progressbar

class LSTM():
    def __init__(self, input_dim, output_dim, is_attention = False, is_dropout = False):
        """
        input_dim: dimension of input data (Tx, n_x)
        output_dim: dimension of output hidden state (Tx, n_a)
        """

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.params = {}

        # a toggle used to decide LSTM_backward
        self.is_backward = False
        self.is_attention = is_attention
        # retrieve dimension
        _, self.n_x = self.input_dim # 256
        _, self.n_a = self.output_dim # 256

        # initialize cell params
        _re_cell_W = func.orthogonal(self.n_a) # 256,256
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
        drop_backward = np.ones(concat.shape)
        if is_dropout:
            concat, drop_backward = act.dropout(concat, level = 0.5)

        ctt = act.tanh(np.matmul(concat, np.transpose(self.params["Wc"])) + self.params["bc"])
        fu = act.sigmoid(np.matmul(concat, np.transpose(self.params["Wu"])) + self.params["bu"])
        ff = act.sigmoid(np.matmul(concat, np.transpose(self.params["Wf"])) + self.params["bf"])
        fo = act.sigmoid(np.matmul(concat, np.transpose(self.params["Wo"])) + self.params["bo"])
        ct = np.multiply(fu, ctt) + np.multiply(ff, c_prev)
        at = np.multiply(fo, act.tanh(ct))

        cache = (concat, ctt, fu, ff, fo, ct, at, a_prev, c_prev, drop_backward)
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

        for t in progressbar.progressbar(range):

            xt = np.atleast_2d(X[t, :])
            at, ct, cache = self.cell_forward(a_prev, c_prev, xt)
            a_prev = at

            c_prev = ct
            self.A.append(at.reshape(-1))
            caches.append(cache)

        return np.array(self.A), caches

    # TODO: check dimension
    def cell_backward(self, dZ, da_next, dc_next, cache_t, Wy, ds_c_next = None):
        """
        backpropagation of 1 cell of post_LSTM---
        ----Parameters----
        dZ: gradient of a = softmax(Z)
        da_next: gradient of hidden state of the cell after current cell
        dc_next: gradient of memory state of the cell after current cell
        cache_t: cache of all values of the current cell (concat, ctt, fu, ff, fo, ct, at, a_prev, c_prev)
        Wy: weight of last layer
        ds_c_next: gradient of context of time-step (t+1)

        ----Return-----
        gradients: a dictionary of gradients of Wf, Wu, Wo, Wctt, bf, bu, bo, bctt
        """
        concat, ctt, fu, ff, fo, ct, at, a_prev, c_prev, d_drop = cache_t
        gradients = {}
        # derivative of Wy and by
        dWy = dZ * at
        dby = dZ

        # derivative of ds at current time-step
        da_t = dZ * Wy
        da = None
        if is_attention:
            da = da_next + da_t + ds_c_next
        else:
            assert(ds_c_next == None)
            da = da_next + da_t

        # derivative of fo at current time-step
        dfo = da * act.tanh(ct) * act.backward_sigmoid(fo)

        # derivative of c
        dc = (da * act.backward_tanh(ct) * fo) + dc_next

        # derivative of ff
        dff = dc * c_prev * act.backward_sigmoid(ff)

        # derivative of fu
        dfu = dc *  ctt * act.backward_sigmoid(fu)

        # derivative of ctt
        dctt = dc * fu * act.backward_tanh(ctt)

        # gate gradients
        dWf = dff * np.transpose(concat)
        dWu = dfu * np.transpose(concat)
        dWo = dfo * np.transpose(concat)
        dWctt = dctt * np.transpose(concat)

        dbf = dff
        dbu = dfu
        dbo = dfo
        dbctt = dctt

        # previous hidden state gradient
        da_prev = self.params["Wf"][:, :self.n_a] * dff * d_drop + self.params["Wu"][:, :self.n_a] * dfu * d_drop + self.params["Wc"][:, :self.n_a] * dctt * d_drop + self.params["Wo"][:, :self.n_a] * dfo * d_drop
        dX = self.params["Wf"][:, self.n_a:] * dff * d_drop + self.params["Wu"][:, self.n_a:] * dfu * d_drop + self.params["Wc"][:, self.n_a:] * dctt * d_drop + self.params["Wo"][:, self.n_a:] * dfo * d_drop
        dc_prev = dc * ff

if __name__ == "__main__":
    input_dim = (10,10,3)
    output_dim = (10,10,5)

    X = np.random.random((10,10,3))
    bi_LSTM = Bidirectional(LSTM(input_dim, output_dim), X[1,:,:])
    A = bi_LSTM.concatLSTM()
    print(A.shape)
