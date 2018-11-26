import numpy as np
import os
import sys
import progressbar
from LSTM import LSTM
from wrapper import Bidirectional
from Regularization import regularization
from attention_model import attention_model
from data_preprocessing import song_preprocessing
from functions import activations as act, helper_func as func
from sklearn.preprocessing import normalize


class model:
    def __init__(self, X, Y, S, Tx, Ty, n_a = 32, n_s = 64, jump_step = 100, epoch = 100):
        self.X = X
        self.Y = Y
        self.S = S
        self.Tx = Tx
        self.Ty = Ty
        self.m = X.shape[0]
        self.n_x = X.shape[2]
        self.n_y = Y.shape[2]
        self.n_a = n_a
        self.n_s = n_s
        self.n_c = self.n_a * 2
        self.hidden_dimension = [64]
        self.layers = {}
        self.jump_step = jump_step
        self.epoch = epoch
        # Wy shape = (n_s,n_y)
        self.Wy = func.xavier((self.n_s, self.n_y))
        self.by = np.zeros((1, self.n_y))



    def forward_propagation_one_ex(self, i):
        print("epoch {}/{}".format(i,self.m))
        X = normalize(self.X[i,:,:], axis = 0)

        pre_LSTM = LSTM((self.Tx, self.n_x), (self.Tx, self.n_a))
        pre_bi_LSTM = Bidirectional(pre_LSTM, X)

        self.layers["pre_bi_LSTM"] = pre_bi_LSTM

        A = pre_bi_LSTM.concatLSTM() # shape = (Tx, 2 * n_a)

        # TODO: dropout A
        A = np.array(act.dropout(A, level=0.5)[0])

        # attention and post_LSTM
        attention = attention_model(self.n_c, A, self.S, self.n_s, self.hidden_dimension)

        self.layers["attention"] = attention

        post_LSTM = LSTM((self.Ty, self.n_c), (self.Ty, self.n_s), is_attention = True, is_dropout = True)

        self.layers["post_LSTM"] = post_LSTM

        start = 0
        end = self.S

        prev_s = np.zeros((1, self.n_s))
        prev_a = np.zeros((1, self.n_s))

        lstm_S = []
        Att_As = []
        Att_caches = []
        Att_alphas = []
        print("Calulating LSTM_S......")
        for t in progressbar.progressbar(range(Ty)):
            alphas, c, _energies, _caches_t, current_A = attention.nn_forward_propagation(prev_s, start, end)
            start = start + jump_step
            end = end + jump_step
            # for backpropagation use
            Att_As.append(current_A)
            Att_caches.append(_caches_t)
            Att_alphas.append(alphas)

            st, at, cache = post_LSTM.cell_forward(prev_s, prev_a, c)
            lstm_S.append(st)
            prev_s = st
            prev_a = at

        # convert lstm_S(list) to lstm_S(np array)
        lstm_S = np.array(lstm_S)
        # TODO: dropout lstm_S
        # lstm_S = act.dropout(lstm_S, level = 0.5)
        # initialize last layer Wy
        # st shape = (1,n_s)
        Y_hat = []
        print("Predicting Y")
        for st in progressbar.progressbar(lstm_S): # st shape = (1, n_s)
            Zy = np.matmul(st, self.Wy) + self.by # shape = (1, n_y)
            yt_hat = act.softmax(Zy)
            Y_hat.append(yt_hat.reshape(-1)) # yt_hat after reshape = (n_y,)

        # Y_hat shape = (Ty, n_y)
        Y_true = np.array(self.Y[i,:,:]) # (Ty, n_y)
        Y_hat = np.array(Y_hat)
        total_lost = 0
        print("Lost....")
        for t in range(Ty):
            lost = func.t_lost(Y_true[t,:], Y_hat[t,:])
            total_lost = total_lost + lost

        total_lost = -(total_lost/Ty) # minimize total_lost = maximize P

        return total_lost, Y_hat, Y_true, lstm_S, Att_As, Att_alphas, Att_caches

    def backward_propagation_one_ex(self, Y_hat, Y_true, lstm_S, Att_As, Att_alphas, Att_caches):

        post_LSTM = self.layers["post_LSTM"]
        attention = self.layers["attention"]
        pre_bi_LSTM = self.layers["pre_bi_LSTM"]

        dL = -(1/Ty)
        # shape (Ty, n_y)
        dZ = dL * (Y_hat - Y_true)
        assert(dZ.shape == (self.Ty, self.n_y))

        # calculate dWy and dby
        dWy = np.matmul(np.transpose(lstm_S.reshape(self.Ty, self.n_s)), dZ)
        dby = np.sum(dZ, axis = 0)
        assert(dWy.shape == (self.n_s, self.n_y) and dby.shape == (1, self.n_y))


        #shape = (Ty, n_s)
        dA = np.matmul(dZ, np.tranpose(self.Wy))

        dict = post_LSTM.backward_propagation(dA, Att_As, Att_caches, Att_alphas, attention)
        attention.update_layer(lr=0.005)
        post_LSTM.update_weight(dict["gradients"], lr=0.005)
        d_AS_list = dict["d_AS_list"]
        pre_bi_LSTM.cell_backpropagation(d_AS_list, self.jump_step, self.Ty)

        return dWy, dby

    def update_weight(self, dWy, dby, lr=0.005):
        self.Wy = self.Wy - lr*dWy
        self.by = self.by - lr*dby

    def train(self):
        print("Starting to train Detector..........")
        for e in range(self.epoch)
            print("Epoch {}/{}".format(e, self.epoch))
            for i in range(m):
                total_lost, Y_hat, Y_true, lstm_S, Att_As, Att_alphas, Att_caches = forward_propagation_one_ex(i)
                print("Total Lost: ", total_lost)
                dWy, dby = self.backward_propagation_one_ex(Y_hat, Y_true, lstm_S, Att_As, Att_alphas, Att_caches)
                self.update_weight(dWy, dby)
