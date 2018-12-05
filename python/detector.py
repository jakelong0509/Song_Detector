import numpy as np
import os
import sys
import progressbar
from model import model
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

    #song_preprocessing.split_song(songs_dir + "/")
    # preprocessing data X.shape = (m, Tx, n_x) | Y.shape = (m, Ty, n_y)
    X, Y = song_preprocessing.preprocessing_data(songs_dir + "/", Tx, Ty)
    model = model(X, Y, S, Tx, Ty, lr = 0.005, n_a = 64, n_s = 128, jump_step = jump_step, epoch = 100, sec = sec, optimizer="Adam")
    model.train()
