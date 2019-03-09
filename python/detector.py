import numpy as np
import os
import sys
import progressbar
import random
import matplotlib.pyplot as plt
from model import model
from pre_song_model import pre_model
from LSTM import LSTM
from wrapper import Bidirectional
from Regularization import regularization
from attention_model import attention_model
from data_preprocessing import song_preprocessing
from functions import activations as act, helper_func as func
from sklearn.preprocessing import normalize, minmax_scale
from scipy.io import wavfile

if __name__ == "__main__":
    sec = 5
    jump_step = 50
    main_dir = os.getcwd()

    # change directory to get songs
    os.chdir("../song_train")
    songs_dir = os.getcwd()

    # get Tx
    Tx, duration = song_preprocessing.get_Tx(songs_dir + "/")
    sec_per_spec = duration / Tx
    S = int(np.round(sec / sec_per_spec))

    # return back to main directory
    os.chdir(main_dir)

    # get Ty
    Ty = song_preprocessing.get_Ty(Tx, S, jump_step)

    # song_preprocessing.split_song(songs_dir + "/")
    # preprocessing data X.shape = (m, Tx, n_x) | Y.shape = (m, Ty, n_y)
    #songs = song_preprocessing.get_songs("../songs")
    songs = ["Everyday", "HitchCock", "Thanh Xuan", "Everyday", "HitchCock", "Thanh Xuan"] # for testing
    songs_test = song_preprocessing.get_songs("../songs_splited")
    X, Y = song_preprocessing.preprocessing_data(songs_dir + "/", Tx, Ty)
    loss = np.random.random(10)


    # reorder training data-------
    order = [0,2,4,1,3,5]
    X_train = np.array([])
    Y_train = np.array([])
    for i in order:
        X_train = np.append(X_train, X[i,:,:])
        Y_train = np.append(Y_train, Y[i,:,:])
    X_train = X_train.reshape(X.shape)
    Y_train = Y_train.reshape(Y.shape)
    #------------------------------------
    folder = "weights" # default folder
    songs_test_ = song_preprocessing.get_songs("../songs")
    # train and test
    model = model(X_train, Y_train, S, Tx, Ty, lr = 0.005, n_a = 128, n_s = 64, jump_step = jump_step, epoch = 1000, sec = sec, optimizer="Adam")
    if sys.argv[1] == "-train":
        model.train(songs)
    elif sys.argv[1] == "-test":
        count = 0
        for s in songs_test:
            print("song: ", s)
            X_predict, duration = song_preprocessing.graph_spectrogram("../songs_splited/"+s)

            if sys.argv[2] == "-f":
                folder = sys.argv[3]

            p_s = model.predict(np.transpose(X_predict), songs_test_, folder)
            if s[:-5] == p_s[:-4]:
                count = count + 1
            print("{}/{}".format(count, len(songs_test)))
    elif sys.argv[1] == "-s":
        print("song: ", str(sys.argv[2]))
        X_predict, duration = song_preprocessing.graph_spectrogram("../songs_splited/"+str(sys.argv[2]))

        if sys.argv[3] == "-f":
            folder = sys.argv[4]


        model.predict(np.transpose(X_predict), songs_test_, folder)
