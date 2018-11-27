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
from optimizers import Adam

class model:
    def __init__(self, X, Y, S, Tx, Ty, lr = 0.1, n_a = 32, n_s = 64, jump_step = 100, epoch = 100, sec = 5, optimizer = None):
        self.X = X
        self.Y = Y
        self.S = S
        self.Tx = Tx
        self.Ty = Ty
        self.lr = lr
        self.m = X.shape[0]
        self.n_x = X.shape[2]
        self.n_y = Y.shape[2]
        self.n_a = n_a
        self.n_s = n_s
        self.n_c = self.n_a * 2
        self.hidden_dimension = [64]
        self.jump_step = jump_step
        self.epoch = epoch
        self.sec = sec
        self.last_layer_hidden_state = None
        self.Att_As = []
        self.Att_caches = []
        self.Att_alphas = []
        # Wy shape = (n_s,n_y)
        self.Wy = func.xavier((self.n_s, self.n_y))
        self.by = np.zeros((1, self.n_y))


        self.pre_LSTM = LSTM("pre_LSTM", (self.Tx, self.n_x), (self.Tx, self.n_a), optimizer = self.optimizer)
        self.pre_bi_LSTM = Bidirectional("pre_bi_LSTM", self.pre_LSTM)
        self.attention = attention_model("attention", self.n_c, self.S, self.n_s, self.hidden_dimension, optimizer = self.optimizer)
        self.post_LSTM = LSTM("post_LSTM", (self.Ty, self.n_c), (self.Ty, self.n_s), is_attention = True, is_dropout = True, optimizer = self.optimizer)



    def forward_propagation_one_ex(self, i):
        """
        description:
            forward propagation for one training example; data x label y
        ---parameter---
        i: index
        """
        X = normalize(self.X[i,:,:], axis = 0)
        A = self.pre_bi_LSTM.concatLSTM(X) # shape = (Tx, 2 * n_a)
        # TODO: dropout A
        A = np.array(act.dropout(A, level=0.5)[0])
        self.attention._A = A
        # attention and post_LSTM
        start = 0
        end = self.S
        prev_s = np.zeros((1, self.n_s))
        prev_a = np.zeros((1, self.n_s))
        lstm_S = []
        print("Calulating LSTM_S......")
        for t in progressbar.progressbar(range(self.Ty)):
            alphas, c, _energies, _caches_t, current_A = self.attention.nn_forward_propagation(prev_s, start, end)
            start = start + jump_step
            end = end + jump_step
            # for backpropagation use
            self.Att_As.append(current_A)
            self.Att_caches.append(_caches_t)
            self.Att_alphas.append(alphas)

            st, at, cache = self.post_LSTM.cell_forward(prev_s, prev_a, c)
            lstm_S.append(st)
            prev_s = st
            prev_a = at

        # convert lstm_S(list) to lstm_S(np array)
        lstm_S = np.array(lstm_S)
        self.last_layer_hidden_state = lstm_S
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

        return total_lost, Y_hat, Y_true

    def backward_propagation_one_ex(self, Y_hat, Y_true, i, lr):
        """
        Description:
            backward propagation for one training example; data x label y
        ----parameter---
        Y_hat: predicted value given training data X
        Y_true: True label value of training data X
        """
        dL = -(1/Ty)
        # shape (Ty, n_y)
        dZ = dL * (Y_hat - Y_true)
        assert(dZ.shape == (self.Ty, self.n_y))
        # calculate dWy and dby
        dWy = np.matmul(np.transpose(self.last_layer_hidden_state.reshape(self.Ty, self.n_s)), dZ)
        dby = np.sum(dZ, axis = 0)
        self.update_weight(dWy, dby, lr)
        assert(dWy.shape == (self.n_s, self.n_y) and dby.shape == (1, self.n_y))
        #shape = (Ty, n_s)
        dS = np.matmul(dZ, np.tranpose(self.Wy))
        d_AS_list = self.post_LSTM.backward_propagation(dS, self.Att_As, self.Att_caches, self.Att_alphas, self.attention)
        self.post_LSTM.update_weight(lr, i)
        self.attention.update_weight(lr, i)
        self.pre_bi_LSTM.cell_backpropagation(d_AS_list, self.jump_step, self.Ty)
        self.pre_bi_LSTM.update_weight(lr, i)


    def update_weight(self, dWy, dby, lr=0.005):
        self.Wy = self.Wy - lr*dWy
        self.by = self.by - lr*dby

    def train(self):
        lr = self.lr
        print("Starting to train Detector..........")
        for e in range(self.epoch)
            print("Epoch {}/{}".format(e, self.epoch))
            decay = lr / e
            lr = lr * (1.0 / (1.0 + decay * e))
            for i in range(m):
                total_lost, Y_hat, Y_true = forward_propagation_one_ex(i)
                print("Total Lost: ", total_lost)
                self.backward_propagation_one_ex(Y_hat, Y_true, i, lr)


    def predict(self, data):
        Tx, _ = data.shape
        assert(Tx >= self.S)
        Ty = song_preprocessing.get_Ty(Tx, self.S, self.jump_step)

        A = self.pre_bi_LSTM.concatLSTM(data)

        start = 0
        end = self.S

        prev_s = np.zeros((1, self.n_s))
        prev_a = np.zeros((1, self.n_s))

        lstm_S = []
        print("Calulating LSTM_S......")
        for t in progressbar.progressbar(range(Ty)):
            alphas, c, _energies, _caches_t, current_A = attention.nn_forward_propagation(prev_s, start, end, data_to_predict = A)
            start = start + self.jump_step
            end = end + self.jump_step

            st, at, cache = post_LSTM.cell_forward(prev_s, prev_a, c)
            lstm_S.append(st)
            prev_s = st
            prev_a = at

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
            print(yt_hat)
            Y_hat.append(yt_hat.reshape(-1)) # yt_hat after reshape = (n_y,)
