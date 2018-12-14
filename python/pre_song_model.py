import numpy as np
import os
import sys
import progressbar
import gc
import pickle
from LSTM import LSTM
from wrapper import Bidirectional
from Regularization import regularization
from attention_model import attention_model
from data_preprocessing import song_preprocessing
from functions import activations as act, helper_func as func
from sklearn.preprocessing import normalize


class pre_model:
    def __init__(self, X, Y, Tx, Ty, lr = 0.005, n_a = 64, epoch = 100, optimizer = None):
        self.X = X
        self.Y = Y
        self.Tx = Tx
        self.Ty = Ty
        self.lr = lr
        self.n_a = n_a
        self.n_x = X.shape[1]
        self.n_y = Y.shape[1]
        self.epoch = epoch
        self.last_layer_hidden_state = None
        # Wy shape = (n_s,n_y)
        self.Wy = func.xavier((self.n_a, self.n_y))
        self.by = np.zeros((1, self.n_y))
        self.optimizer = optimizer
        self.s_weight = 0
        self.s_bias = 0
        self.v_weight = 0
        self.v_bias = 0
        self.TRAINING_THRESHOLD = 0
        self._params = {"Wy": self.Wy, "by": self.by}

        self.pre_LSTM = LSTM("pre_LSTM", (self.Tx, self.n_x), (self.Tx, self.n_a), optimizer = optimizer, is_dropout = True)

    def forward_propagation_one_ex(self, e):
        """
        description:
            forward propagation for one training example; data x label y
        ---parameter---
        i: index
        """
        # self.gradient_checking()

        A = self.pre_LSTM.forward_propagation(self.X) # shape = (Tx, 2 * n_a)
        self.last_layer_hidden_state = A
        # TODO: dropout A
        #A = np.array(act.dropout(A, level=0.8)[0])

        # TODO: dropout lstm_S
        # lstm_S = act.dropout(lstm_S, level = 0.5)
        # initialize last layer Wy
        # st shape = (1,n_s)
        Y_hat = []
        print("Predicting Y")
        for t in progressbar.progressbar(range(self.Ty)): # st shape = (1, n_s)
            Zy = np.matmul(np.atleast_2d(A[t,:]), self._params["Wy"]) + self._params["by"] # shape = (1, n_y)
            yt_hat = act.softmax(Zy)
            Y_hat.append(yt_hat.reshape(-1)) # yt_hat after reshape = (n_y,)

        # Y_hat shape = (Ty, n_y)
        Y_true = np.array(self.Y) # (Ty, n_y)
        Y_hat = np.array(Y_hat)
        total_lost = 0
        print("Lost....")
        for t in range(self.Ty):
            lost = func.t_lost(Y_true[t,:], Y_hat[t,:])
            total_lost = total_lost + lost

        total_lost = (total_lost/self.Ty)

        return total_lost, Y_hat, Y_true

    def backward_propagation_one_ex(self, Y_hat, Y_true, e, lr):
        """
        Description:
            backward propagation for one training example; data x label y
        ----parameter---
        Y_hat: predicted value given training data X
        Y_true: True label value of training data X
        """
        # dL = (1/self.Ty)
        # shape (Ty, n_y)
        dZ = (Y_hat - Y_true)
        assert(dZ.shape == (self.Ty, self.n_y))
        # calculate dWy and dby
        dWy = np.matmul(np.transpose(self.last_layer_hidden_state.reshape(self.Ty, self.n_a)), dZ)
        dby = np.atleast_2d(np.sum(dZ, axis = 0))
        self.update_weight(dWy, dby, e, lr, optimizer = self.optimizer)

        assert(dWy.shape == (self.n_a, self.n_y) and dby.shape == (1, self.n_y))
        #shape = (Ty, n_a)
        dA = np.matmul(dZ, np.transpose(self._params["Wy"]))
        self.pre_LSTM.backward_propagation(dA)

    def update_weight(self, dWy, dby, i ,lr=0.005, optimizer = None, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):

        i = i + 1
        lr = lr * np.sqrt(1-beta2**i)/(1-beta1**i)
        s_corrected_weight = None
        s_corrected_bias = None
        v_corrected_weight = None
        v_corrected_bias = None
        if optimizer == "Adam":
            self.s_weight = beta2 * self.s_weight + (1 - beta2) * (dWy ** 2)
            s_corrected_weight = self.s_weight / (1 - beta2**i)
            self.s_bias = beta2 * self.s_bias + (1 - beta2) * (dby ** 2)
            s_corrected_bias = self.s_bias / (1 - beta2**i)

            self.v_weight = beta1 * self.v_weight + (1 - beta1) * dWy
            v_corrected_weight = self.v_weight / (1 - beta1**i)
            self.v_bias = beta1 * self.v_bias + (1 - beta1) * dby
            v_corrected_bias = self.v_bias / (1 - beta1**i)

            self.Wy = self.Wy - lr*(v_corrected_weight/(np.sqrt(s_corrected_weight) + eps))
            self.by = self.by - lr*(v_corrected_bias/(np.sqrt(s_corrected_bias) + eps))
        else:
            self.Wy = self.Wy - lr*dWy
            self.by = self.by - lr*dby

        self._params["Wy"] = self.Wy
        self._params["by"] = self.by

        self.save_weights()

    def save_weights(self):
        with open("weights_pre_song/predict_layer.pickle", "wb") as f:
            pickle.dump(self._params, f, protocol = pickle.HIGHEST_PROTOCOL)

    def train(self, songs):
        lr = self.lr
        print("Starting to train Detector..........")
        for e in range(self.epoch):
            print("Epoch {}/{}".format(e, self.epoch))


            total_lost, Y_hat, Y_true = self.forward_propagation_one_ex(e)
            print("Total Lost: ", total_lost)
            self.backward_propagation_one_ex(Y_hat, Y_true, e, lr)
