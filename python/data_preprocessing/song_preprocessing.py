import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import os
from pydub import AudioSegment


# return song list and number of song
def get_songs(dir):
    songs = None;
    song_no = 0;
    for(dirpath, dirnames, filenames) in os.walk(dir):
        songs = filenames
        song_no = len(filenames)

    return songs, song_no

# return Tx of longest song
def get_Tx(dir):
    Tx = 0
    songs, _ = get_songs(dir)
    for s in songs:
        Tx_temp = graph_spectrogram(dir+s).shape[1]
        if(Tx_temp > Tx):
            Tx = Tx_temp
    return Tx

# create label(y) for training
def set_labels(y_oh, Ty, index): # y.shape = (Ty, n_y)
    y = np.repeat(y_oh, Ty, axis=0)
    return y

# return index of song base on song name
def get_y_index(dir, song_name):
    songs, song_no = get_songs(dir)
    songs_indices = []
    y_index = 0;
    count = 0;
    for s in songs:
        if(song_name.lower() == s.lower()):
            y_index = count
            break
        count = count + 1
    return y_index

# convert song to one-hot representation
def to_one_hot(n_y, y_index):
    y_oh = np.zeros((1,n_y), dtype = int)
    y_oh[:,y_index] = 1
    return y_oh

def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    nfft = 200
    fs = 8000
    noverlap = 120
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx

def preprocessing_data(dir, Tx, Ty):
    songs, n_y = get_songs(dir)
    y_indexes = []
    m = n_y
    x = []
    y = []
    for s in songs:
        x_temp = graph_spectrogram(dir+s)
        y_indexes.append(get_y_index(dir, s))
        if(x_temp.shape[1] < Tx):
            missing = Tx - x_temp.shape[1]
            zeros = np.repeat(np.zeros((101,1)), missing, axis=1) # (101,missing)
            x_temp = np.append(x_temp, zeros, axis = 1)

        x.append(x_temp)
    x = np.swapaxes(x, 1,2) # x.shape = (m, Tx, n_x)

    for i in y_indexes:
        y_oh = to_one_hot(n_y, i)
        y_ohs = set_labels(y_oh, Ty, i)
        y.append(y_ohs)
    y = np.array(y)
    return x,y
