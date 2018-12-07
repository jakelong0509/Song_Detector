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


class model:
    def __init__(self, X, Y, S, Tx, Ty, lr = 0.005, n_a = 64, n_s = 32, jump_step = 100, epoch = 100, sec = 5, optimizer = None):
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
        self.hidden_dimension = [64,10]
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
        self.optimizer = optimizer
        self.s_weight = 0
        self.s_bias = 0
        self.v_weight = 0
        self.v_bias = 0
        self.TRAINING_THRESHOLD = 0
        self._params = {"Wy": self.Wy, "by": self.by}

        self.pre_LSTM = LSTM("pre_LSTM", (self.Tx, self.n_x), (self.Tx, self.n_a), optimizer = optimizer, is_dropout = True)
        self.pre_bi_LSTM = Bidirectional("pre_bi_LSTM", self.pre_LSTM)
        self.attention = attention_model("attention", self.n_c, self.S, self.n_s, self.n_c, self.hidden_dimension, optimizer = optimizer)
        self.post_LSTM = LSTM("post_LSTM", (self.Ty, self.n_c), (self.Ty, self.n_s), is_attention = True, optimizer = optimizer)



    def forward_propagation_one_ex(self, i, e):
        """
        description:
            forward propagation for one training example; data x label y
        ---parameter---
        i: index
        """
        # self.gradient_checking()
        X = normalize(self.X[i,:,:], axis = 1)
        X = act.relu(X)
        A = self.pre_bi_LSTM.concatLSTM(X) # shape = (Tx, 2 * n_a)
        # TODO: dropout A
        #A = np.array(act.dropout(A, level=0.8)[0])

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
            start = start + self.jump_step
            end = end + self.jump_step

            # for backpropagation use ***** this step take 30% of RAM in total *******
            self.Att_As.append(current_A)
            self.Att_caches.append(_caches_t)
            self.Att_alphas.append(alphas)

            st, at, cache = self.post_LSTM.cell_forward(prev_s, prev_a, c)
            lstm_S.append(st)
            prev_s = st
            prev_a = at


        # convert lstm_S(list) to lstm_S(np array)
        lstm_S = np.array(lstm_S).reshape((self.Ty, self.n_s))

        self.last_layer_hidden_state = lstm_S
        del lstm_S
        # TODO: dropout lstm_S
        # lstm_S = act.dropout(lstm_S, level = 0.5)
        # initialize last layer Wy
        # st shape = (1,n_s)
        Y_hat = []
        print("Predicting Y")
        for t in progressbar.progressbar(range(self.Ty)): # st shape = (1, n_s)
            Zy = np.matmul(np.atleast_2d(self.last_layer_hidden_state[t,:]), self._params["Wy"]) + self._params["by"] # shape = (1, n_y)
            yt_hat = act.softmax(Zy)
            print(yt_hat)
            Y_hat.append(yt_hat.reshape(-1)) # yt_hat after reshape = (n_y,)

        # Y_hat shape = (Ty, n_y)
        Y_true = np.array(self.Y[i,:,:]) # (Ty, n_y)
        Y_hat = np.array(Y_hat)
        total_lost = 0
        print("Lost....")
        for t in range(self.Ty):
            lost = func.t_lost(Y_true[t,:], Y_hat[t,:])
            total_lost = total_lost + lost

        total_lost = (total_lost/self.Ty)

        return total_lost, Y_hat, Y_true

    def backward_propagation_one_ex(self, Y_hat, Y_true, i, e, lr):
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
        dWy = np.matmul(np.transpose(self.last_layer_hidden_state.reshape(self.Ty, self.n_s)), dZ)
        dby = np.atleast_2d(np.sum(dZ, axis = 0))
        self.update_weight(dWy, dby, e, lr, optimizer = self.optimizer)

        assert(dWy.shape == (self.n_s, self.n_y) and dby.shape == (1, self.n_y))
        #shape = (Ty, n_s)

        dS = np.matmul(dZ, np.transpose(self.Wy))
        d_AS_list = self.post_LSTM.backward_propagation(dS, self.Att_As, self.Att_caches, self.Att_alphas, self.attention)
        self.post_LSTM.update_weight(lr, e)
        self.attention.update_weight(lr, e)

        self.Att_As = []
        self.Att_caches = []
        self.Att_alphas = []


        self.pre_bi_LSTM.cell_backpropagation(d_AS_list, self.jump_step, self.Ty, self.Tx)
        self.pre_bi_LSTM.update_weight(lr, e)



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

    def train(self, songs):
        lr = self.lr
        print("Starting to train Detector..........")
        for e in range(self.epoch):
            print("Epoch {}/{}".format(e, self.epoch))
            for i in progressbar.progressbar(range(self.m)):

                total_lost, Y_hat, Y_true = self.forward_propagation_one_ex(i, e)
                print("Total Lost: ", total_lost)
                self.backward_propagation_one_ex(Y_hat, Y_true, i, e, lr)
                self.predict(self.X[i,:,:], songs)

    def save_weights(self):
        with open("weights/predict_layer.pickle", "wb") as f:
            pickle.dump(self._params, f, protocol = pickle.HIGHEST_PROTOCOL)

    def predict(self, data, songs):
        Tx, n_x = data.shape
        assert(Tx >= self.S)

        pre_LSTM = LSTM("pre_LSTM", (Tx, n_x), (Tx, self.n_a), optimizer = self.optimizer)
        pre_bi_LSTM = Bidirectional("pre_bi_LSTM", pre_LSTM)
        attention = attention_model("attention", self.n_c, self.S, self.n_s, self.n_c, self.hidden_dimension, optimizer = self.optimizer)
        post_LSTM = LSTM("post_LSTM", (self.Ty, self.n_c), (self.Ty, self.n_s), is_attention = True, optimizer = self.optimizer)

        LSTM_forward_params = pickle.load(open("weights/biDirectional_pre_LSTM_forward.pickle", "rb"))
        LSTM_backward_params = pickle.load(open("weights/biDirectional_pre_LSTM_backward.pickle", "rb"))
        attention_params = pickle.load(open("weights/attention.pickle", "rb"))
        post_LSTM_params = pickle.load(open("weights/post_LSTM.pickle", "rb"))
        params = pickle.load(open("weights/predict_layer.pickle", "rb"))

        pre_bi_LSTM.forward._params = LSTM_forward_params
        pre_bi_LSTM.backward._params = LSTM_backward_params
        attention._params = attention_params
        post_LSTM._params = post_LSTM_params

        Ty = song_preprocessing.get_Ty(Tx, self.S, self.jump_step)
        data = normalize(data, axis=1)
        
        A = pre_bi_LSTM.concatLSTM(data)
        attention._A = A
        start = 0
        end = self.S

        prev_s = np.zeros((1, self.n_s))
        prev_a = np.zeros((1, self.n_s))

        lstm_S = []
        print("Calulating LSTM_S......")
        for t in progressbar.progressbar(range(Ty)):
            alphas, c, _energies, _caches_t, current_A = attention.nn_forward_propagation(prev_s, start, end)
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

        y_predict = 0
        print("Predicting Y")
        for t in progressbar.progressbar(range(Ty)): # st shape = (1, n_s)
            Zy = np.matmul(np.atleast_2d(lstm_S[t,:]), params["Wy"]) + params["by"] # shape = (1, n_y)
            yt_hat = act.softmax(Zy)
            y_predict = y_predict + yt_hat
        y_predict = y_predict / Ty
        print(y_predict)
        index = np.argmax(y_predict)
        print(songs[index])

    def gradient_checking(self, dby, dWy, i, eps = 1e-4):
        model_vec, model_keys_shape = func.dictionary_to_vector(self._params)
        LSTM_forward_vec, LSTM_forward_keys_shape = func.dictionary_to_vector(self.pre_bi_LSTM.forward._params)
        LSTM_backward_vec, LSTM_backward_keys_shape = func.dictionary_to_vector(self.pre_bi_LSTM.backward._params)
        attention_vec, attention_keys_shape = func.dictionary_to_vector(self.attention._params)
        post_LSTM_vec, post_LSTM_keys_shape = func.dictionary_to_vector(self.post_LSTM._params)

        params_vector = np.concatenate([model_vec, LSTM_forward_vec, LSTM_backward_vec, attention_vec, post_LSTM_vec])
        remain_vector = None
        model_dict = {"dby": dby, "dWy": dWy}
        model_grads, model_grads_keys_shape = func.dictionary_to_vector(model_dict)
        LSTM_forward_grads, LSTM_forward_grads_keys_shape = func.dictionary_to_vector(self.pre_bi_LSTM.forward.gradients)
        LSTM_backward_grads, LSTM_backward_grads_keys_shape = func.dictionary_to_vector(self.pre_bi_LSTM.backward.gradients)
        attention_grads, attention_grads_keys_shape = func.dictionary_to_vector(self.attention.gradients_layer)
        post_LSTM_grads, post_LSTM_grads_keys_shape = func.dictionary_to_vector(self.post_LSTM.gradients)

        grads_vector = np.concatenate([model_grads, LSTM_forward_grads, LSTM_backward_grads, attention_grads, post_LSTM_grads])

        num_parameters = params_vector.shape[0]
        J_plus = np.zeros((num_parameters, 1))
        J_minus = np.zeros((num_parameters, 1))
        gradapprox = np.zeros((num_parameters, 1))
        for n in range(num_parameters):
            print("{}/{}".format(n,num_parameters))
            thetaplus = np.copy(params_vector)
            thetaplus[n] = thetaplus[n] + eps
            remain_vector, model_params = func.vector_to_dictionary(thetaplus, model_keys_shape)
            remain_vector, LSTM_forward_params = func.vector_to_dictionary(remain_vector, LSTM_forward_keys_shape)
            remain_vector, LSTM_backward_params = func.vector_to_dictionary(remain_vector, LSTM_backward_keys_shape)
            remain_vector, attention_params = func.vector_to_dictionary(remain_vector, attention_keys_shape)
            remain_vector, post_LSTM_params = func.vector_to_dictionary(remain_vector, post_LSTM_keys_shape)

            self._params = model_params
            self.pre_bi_LSTM.forward._params = LSTM_forward_params
            self.pre_bi_LSTM.backward._params = LSTM_backward_params
            self.attention._params = attention_params
            self.post_LSTM._params = post_LSTM_params

            J_plus[n], _, _ = self.forward_propagation_one_ex(i)


            thetaminus = np.copy(params_vector)
            thetaminus[n] = thetaminus[n] + eps
            remain_vector, model_params = func.vector_to_dictionary(thetaminus, model_keys_shape)
            remain_vector, LSTM_forward_params = func.vector_to_dictionary(remain_vector, LSTM_forward_keys_shape)
            remain_vector, LSTM_backward_params = func.vector_to_dictionary(remain_vector, LSTM_backward_keys_shape)
            remain_vector, attention_params = func.vector_to_dictionary(remain_vector, attention_keys_shape)
            remain_vector, post_LSTM_params = func.vector_to_dictionary(remain_vector, post_LSTM_keys_shape)

            self._params = model_params
            self.pre_bi_LSTM.forward._params = LSTM_forward_params
            self.pre_bi_LSTM.backward._params = LSTM_backward_params
            self.attention._params = attention_params
            self.post_LSTM._params = post_LSTM_params

            J_minus[n], _, _ = self.forward_propagation_one_ex(i)

            gradapprox[n] = (J_plus[n] - J_minus[n]) / (2 * eps)


        numerator = np.linalg.norm(grads_vector - gradapprox)
        demoninator = np.linalg.norm(grads_vector) + np.linalg.norm(gradapprox)
        difference = numerator / demoninator

        if difference > 1e-7:
            print("Wrong")
        else:
            print("Right")
