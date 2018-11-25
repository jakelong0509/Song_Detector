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

if __name__ == "__main__":
    sec = 5
    jump_step = 200
    main_dir = os.getcwd()

    # change directory to get songs
    os.chdir("../songs")
    songs_dir = os.getcwd()

    # get Tx
    Tx, duration = song_preprocessing.get_Tx(songs_dir + "/")
    sec_per_spec = duration / Tx
    S = int(np.round(sec / sec_per_spec))

    # return back to main directory
    os.chdir(main_dir)

    # get Ty
    Ty = song_preprocessing.get_Ty(Tx, S, jump_step)

    # preprocessing data X.shape = (m, Tx, n_x) | Y.shape = (m, Ty, n_y)
    X, Y = song_preprocessing.preprocessing_data(songs_dir + "/", Tx, Ty)

    m = X.shape[0]

    n_x = X.shape[2]
    n_y = Y.shape[2]
    n_a = 32
    X = normalize(X[0,:,:], axis = 0)

    pre_LSTM = LSTM((Tx, n_x), (Tx, n_a))
    pre_bi_LSTM = Bidirectional(pre_LSTM, X)
    A = pre_bi_LSTM.concatLSTM() # shape = (Tx, 2 * n_a)

    # TODO: dropout A
    A = np.array(act.dropout(A, level=0.5)[0])

    # attention and post_LSTM
    n_s = 64
    n_c = n_a * 2
    post_LSTM = LSTM((Ty, n_c), (Ty, n_s), is_attention = True, is_dropout = True, is_lastlayer = True)
    start = 0
    end = S
    prev_s = np.zeros((1, n_s))
    prev_a = np.zeros((1, n_s))
    hidden_dimension = [64]
    lstm_S = []
    attention = attention_model(A, S, n_s, hidden_dimension)
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
    # Wy shape = (n_s,n_y)
    # st shape = (1,n_s)
    Wy = func.xavier((n_s, n_y))
    by = np.zeros((1, n_y))
    Y_hat = []
    print("Predicting Y")
    for st in progressbar.progressbar(lstm_S): # st shape = (1, n_s)
        Zy = np.matmul(st, Wy) + by # shape = (1, n_y)
        yt_hat = act.softmax(Zy)
        Y_hat.append(yt_hat.reshape(-1)) # yt_hat after reshape = (n_y,)

    # Y_hat shape = (Ty, n_y)
    Y_true = np.array(Y[0,:,:]) # (Ty, n_y)
    Y_hat = np.array(Y_hat)
    total_lost = 0
    print("Lost....")
    for t in range(Ty):
        lost = func.t_lost(Y_true[t,:], Y_hat[t,:])
        total_lost = total_lost + lost

    total_lost = -(total_lost/Ty) # minimize total_lost = maximize P
    print(total_lost)


    dL = -(1/Ty)
    gradients = post_LSTM.lastlayer_backpropagation(dL, Y_true, Y_hat, Wy, Att_As, Att_caches, Att_alphas, attention)
    print(gradients["dWy"])
