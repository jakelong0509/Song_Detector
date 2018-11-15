import numpy as np
import os
import sys

from LSTM import LSTM
from wrapper import Bidirectional
from Regularization import regularization
from attention_model import *
from data_preprocessing import song_preprocessing
from functions import activations as act, helper_func as func

if __name__ == "__main__":
    sec = 5
    main_dir = os.getcwd()

    # change directory to get songs
    os.chdir("../songs")
    songs_dir = os.getcwd()

    # get Tx
    Tx, duration = song_preprocessing.get_Tx(songs_dir + "/")
    sec_per_spec = duration / Tx
    S = np.round(sec / sec_per_spec)

    # return back to main directory
    os.chdir(main_dir)

    # get Ty
    Ty = song_preprocessing.get_Ty(Tx, S)

    # preprocessing data X.shape = (m, Tx, n_x) | Y.shape = (m, Ty, n_y)
    X, Y = song_preprocessing.preprocessing_data(songs_dir + "/", Tx, Ty)
    m = X.shape[0]

    n_x = X.shape[2]
    n_a = 128

    pre_LSTM = LSTM((Tx, n_x), (Tx, n_a))
    pre_bi_LSTM = Bidirectional(pre_LSTM, X[0,:,:])
    A = pre_bi_LSTM.concatLSTM()
    print(A.shape)
