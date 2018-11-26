import numpy as np
import os
import sys
from functions import activations as act, helper_func as func
from wrapper import Bidirectional
import progressbar

class LSTM():
    def __init__(self, name, input_dim, output_dim, is_attention = False, is_dropout = False):
        """
        input_dim: dimension of input data (Tx, n_x)
        output_dim: dimension of output hidden state (Tx, n_a)
        """

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.params = {}
        self.caches = []
        self.gradients = None
        self.name = name
        # a toggle used to decide LSTM_backward
        self.is_backward = False
        self.is_attention = is_attention
        self.is_dropout = is_dropout
        # retrieve dimension
        self.T, self.n_x = self.input_dim
        _, self.n_a = self.output_dim

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
        a_prev: hidden state of previous cell (1, n_a)
        c_prev: cell memory of previous cell (1, n_a)
        xt: current data (1, n_x)
        """
        concat = np.concatenate((a_prev, xt), axis = 1) # shape = (1, n_a + n_x)
        d_ax_drop = np.ones(concat.shape)
        d_c_drop = np.ones(concat.shape)
        if self.is_dropout:
            concat, d_ax_drop = act.dropout(concat, level = 0.5)
            c_prev, d_c_drop = act.dropout(c_prev, level = 0.5)

        ctt = act.tanh(np.matmul(concat, np.transpose(self.params["Wc"])) + self.params["bc"])
        fu = act.sigmoid(np.matmul(concat, np.transpose(self.params["Wu"])) + self.params["bu"])
        ff = act.sigmoid(np.matmul(concat, np.transpose(self.params["Wf"])) + self.params["bf"])
        fo = act.sigmoid(np.matmul(concat, np.transpose(self.params["Wo"])) + self.params["bo"])
        ct = np.multiply(fu, ctt) + np.multiply(ff, c_prev)
        at = np.multiply(fo, act.tanh(ct)) # shape = (1,n_a)

        cache = (concat, ctt, fu, ff, fo, ct, at, a_prev, c_prev, d_ax_drop, d_c_drop)
        self.caches.append(cache)
        return at, ct, cache

    def forward_propagation(self, X):
        """
        X: data of 1 song row (Tx, n_x)
        -----------------------------
        backward case:
            a_prev = a_next
            c_prev = c_next
        """
        Tx, n_x = X.shape
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


        return np.array(self.A)

    # TODO: check dimension
    def cell_backward(self, da_t, da_next, dc_next, cache_t, ds_c_next = None):
        """
        backpropagation of 1 cell of LSTM---
        ----Parameters----
        dZ: gradient of a = softmax(Z) at time step t (1,n_y)
        da_next: gradient of hidden state of the cell after current cell
        dc_next: gradient of memory state of the cell after current cell
        cache_t: cache of all values of the current cell (concat, ctt, fu, ff, fo, ct, at, a_prev, c_prev)
        Wy: weight of last layer
        ds_c_next: gradient of context of time-step (t+1)

        ----Return-----
        gradients: a dictionary of gradients of Wf, Wu, Wo, Wctt, bf, bu, bo, bctt
        """
        concat, ctt, fu, ff, fo, ct, at, a_prev, c_prev, d_ax_drop, d_c_drop = cache_t
        gradients_t = {}

        # derivative of ds at current time-step
        # da_t = None
        # if da_t_calced == None:
        #     da_t = np.matmul(dZ, np.transpose(Wy)) # shape = (1,n_a)
        # else:
        #     da_t = da_t_calced
        assert(da_t.shape == (1, self.n_a))

        da = None
        if self.is_attention:
            da = da_next + da_t + ds_c_next
        else:
            assert(ds_c_next == None)
            da = da_next + da_t

        # derivative of fo at current time-step
        # shape = (1,n_a)
        dfo = da * act.tanh(ct) * act.backward_sigmoid(fo)
        assert(dfo.shape == (1, self.n_a))
        # derivative of c
        # shape = (1,n_a)
        dc = (da * act.backward_tanh(ct) * fo) + dc_next
        assert(dc.shape == (1, self.n_a))
        # derivative of ff
        # shape = (1,n_a)
        dff = dc * c_prev * act.backward_sigmoid(ff)
        assert(dff.shape == (1, self.n_a))
        # derivative of fu
        # shape = (1,n_a)
        dfu = dc *  ctt * act.backward_sigmoid(fu)
        assert(dfu.shape == (1, self.n_a))
        # derivative of ctt
        # shape = (1,n_a)
        dctt = dc * fu * act.backward_tanh(ctt)
        assert(dctt.shape == (1, self.n_a))
        # gate gradients weight
        # shape = (n_a, n_a + n_x)
        dWf = np.matmul(np.transpose(dff), concat)
        assert(dWf.shape == (self.n_a, self.n_a + self.n_x))

        dWu = np.matmul(np.transpose(dfu), concat)
        assert(dWu.shape == (self.n_a, self.n_a + self.n_x))

        dWo = np.matmul(np.transpose(dfo), concat)
        assert(dWo.shape == (self.n_a, self.n_a + self.n_x)))

        dWctt = np.matmul(np.transpose(dctt), concat)
        assert(dWctt.shape == (self.n_a, self.n_a + self.n_x)))

        # gate gradients bias
        # shape = (1, n_a)
        dbf = dff
        dbu = dfu
        dbo = dfo
        dbctt = dctt

        # previous hidden state gradient
        # da_prev shape = (1,n_a)
        # dc_prev shape = (1,n_a)
        # dX shape = (1,n_x)
        da_prev = (np.matmul(dff, self.params["Wf"][:, :self.n_a]) + np.matmul(dfu, self.params["Wu"][:, :self.n_a]) + np.matmul(dctt, self.params["Wc"][:, :self.n_a]) + np.matmul(dfo, self.params["Wo"][:, :self.n_a])) * d_ax_drop[:, :self.n_a]
        dX = (np.matmul(dff, self.params["Wf"][:, self.n_a:]) + np.matmul(dfu, self.params["Wu"][:, self.n_a:]) + np.matmul(dctt, self.params["Wc"][:, self.n_a:]) + np.matmul(dfo, self.params["Wo"][:, self.n_a:])) * d_ax_drop[:, self.n_a:]
        dc_prev = (dc * ff) * d_c_drop

        gradients_t = {"dWf": dWf, "dWu": dWu, "dWo": dWo, "dWctt": dWctt, "dbf": dbf, "dbu": dbu, "dbo": dfo, "dbctt": dbctt}



        return gradients_t, dX, da_prev, dc_prev


    def backward_propagation(self, dA, Att_As = None, Att_caches = None, Att_alphas = None, attention = None):
        """
        ----backpropagation for LSTM layer--
        Parameters:
            dA: gradient of hidden state (Tx, n_a) or (Ty, n_s)
            attention: attention layer object
            Att_As: Hidden state of pre_LSTM
            Att_caches: cache value of attention model
            Att_alphas: alphas of attention model

        returns:
            gradients: weight and bias gradients of entire layer (list)
        """
        if not self.is_attention:
            assert(Att_As == None and Att_caches == None and Att_alphas == None and attention == None)

        # gradient of Z where Y_hat = g(Z) - softmax
        da_next_2 = np.zeros((1,self.n_a))
        dc_next = np.zeros((1,self.n_a))

        ds_c_next = None
        if self.is_attention:
            ds_c_next = np.zeros((1,self.n_a))

        gradients = []
        d_AS_list = []
        dict = {}
        print("Calculating Gradient......")
        for t in progressbar.progressbar(reversed(range(self.T))):
            gradients_t, dX, da_prev, dc_prev = self.cell_backward(np.atleast_2d(dA[t,:]), da_next_2, dc_next, self.caches[t], ds_c_next)
            da_next_2 = da_prev
            dc_next = dc_prev
            gradients.append(gradients_t)
            if self.is_attention:
                # TODO: attention model cell_backpropagation to calc ds_c_prev [ds_c_next = ds_c_prev]
                ds_c_next, d_AS, att_gradients_t = attention.nn_backward_propagation(dX, Att_alphas[t], Att_As[t], Att_caches[t])

                # run multithreading to calc attention model grads at time step t
                attention.cell_update_gradient_t(att_gradients_t, 8)

                # append list d_AS to list d_AS_list
                d_AS_list.append(d_AS) # Ty -> 1
        dict["gradients"] = gradients
        if self.is_attention:
            dict["d_AS_list"] = np.flip(d_AS_list, axis=0) # flip 1->Ty
        return dict

    # def normal_backpropagation(self, dA):
    #     da_prev = np.zeros((1, n_a_normal))
    #     dc_prev = np.zeros((1, n_a_normal))
    #     gradients = []
    #     print("Calculating Gradient......")
    #     for t in progressbar.progressbar(reversed(range(Tx))):
    #         gradients_t, dX, da_prev, dc_prev = self.cell_backward(da_next, dc_next, cache_t, np.atleast_2d(dA[t,:]))
    #         gradients.append(gradients_t)
    #
    #     return gradients;

    def update_gradient(self, gradients):
        """
        ----parameters-----
        gradients: a list of gradient dictionary at time step t in Ty
        """
        grads = {k: np.zeros_like(v) for k,v in gradients[0].items()}
        for grad in gradients:
            for k in grad.keys():
                grads[k] = grads[k] + grad[k]

        self.gradients = grads

    def update_weight(self, gradients, lr):

        # run update_gradient to update self.gradients
        self.update_gradient(gradients)

        # update weight and bias of ctt gate
        self.params["Wc"] = self.params["Wc"] - lr*self.gradients["dWctt"]
        self.params["bc"] = self.params["bc"] - lr*self.gradients["dbctt"]

        # update weight and bias of f gate
        self.params["Wf"] = self.params["Wf"] - lr*self.gradients["dWf"]
        self.params["bf"] = self.params["bf"] - lr*self.gradients["dbf"]

        # update weight and bias of o gate
        self.params["Wo"] = self.params["Wo"] - lr*self.gradients["dWo"]
        self.params["bo"] = self.params["bo"] - lr*self.gradients["dbo"]

        # update weight and bias of u gate
        self.params["Wu"] = self.params["Wu"] - lr*self.gradients["dWu"]
        self.params["bu"] = self.params["bu"] - lr*self.gradients["dbu"]



if __name__ == "__main__":
    input_dim = (10,10,3)
    output_dim = (10,10,5)

    X = np.random.random((10,10,3))
    bi_LSTM = Bidirectional(LSTM(input_dim, output_dim), X[1,:,:])
    A = bi_LSTM.concatLSTM()
    print(A.shape)
