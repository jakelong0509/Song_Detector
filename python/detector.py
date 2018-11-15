import numpy as np
import os
import sys

from LSTM import LSTM
from wrapper import Bidirectional
from Regularization import regularization
from attention_model import attention_model
from data_preprocessing import song_preprocessing
from functions import activations as act, helper_func as func

if __name__ == "__main__":
    sec = 5
    jump_step = 100
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
    n_a = 64

    pre_LSTM = LSTM((Tx, n_x), (Tx, n_a))
    pre_bi_LSTM = Bidirectional(pre_LSTM, X[0,:,:])
    A = pre_bi_LSTM.concatLSTM() # shape = (Tx, 2 * n_a)

    # TODO: dropout A

    # attention and post_LSTM
    n_s = 128
    n_c = n_a * 2
    post_LSTM = LSTM((Ty, n_c), (Ty, n_s))
    start = 0
    end = S
    prev_s = np.zeros((1, n_s))
    prev_a = np.zeros((1, n_s))
    hidden_dimension = [64]
    lstm_S = []
    attentions = []
    for t in range(Ty):
        attention = attention_model(A[start:end,:], prev_s, hidden_dimension)
        attentions.append(attention)
        start = start + jump_step
        end = end + jump_step

        alphas, c, _energies, _caches_t = attention.nn_forward_propagation()
        st, at, cache = post_LSTM.cell_forward(prev_s, prev_a, c)
        lstm_S.append(st)
        prev_s = st
        prev_a = at

    print(np.array(lstm_S).shape)
    # TODO: dropout lstm_S

    # initialize last layer Wy
    # Wy shape = ()
    # st shape = (1,n_s)
    Wy = func.xavier((n_s, n_y))
    by = np.zeros((1, n_y))
    Y_hat = []
    for st in lstm_S:
        Zy = np.matmul(st, Wy) + by # shape = (1, n_y)
        yt_hat = act.softmax(Zy)
        Y_hat.append(yt_hat)

    # Y_hat shape = (Ty, 1, n_y)
    print(A.shape)
