import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import os
import sys
from pydub import AudioSegment

n_y = 3
# return song list and number of song
def get_songs(dir):
    songs = None;
    for(dirpath, dirnames, filenames) in os.walk(dir):
        songs = filenames
    songs.sort()
    return songs



# return Tx of longest song
def get_Tx(dir):
    Tx = 0
    duration = 0
    songs = get_songs(dir)
    for s in songs:
        pxx_temp, duration_temp = graph_spectrogram(dir+s)
        Tx_temp = pxx_temp.shape[1]
        if(Tx_temp > Tx):
            Tx = Tx_temp
            duration = duration_temp
    return Tx, duration

# return Ty
def get_Ty(Tx, S, jump_step = 100):
    window = S
    count = 0;
    while (window <= Tx):
        count = count + 1;
        window = window + jump_step
    Ty = count
    return Ty

# create label(y) for training
def set_labels(y_oh, Ty, index): # y.shape = (Ty, n_y)
    y = np.repeat(y_oh, Ty, axis=0)
    return y

# return index of song base on song name
def get_y_index(dir, song_name):
    songs = get_songs(dir)
    songs_indices = []
    y_index = 0;
    count = 0;
    for s in songs:
        if(song_name[:-5].lower() == s[:-4].lower()):
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
    duration = np.round(data.shape[0]/rate)

    nfft = 200
    fs = 8000
    noverlap = 120
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx, duration

def preprocessing_data(dir, Tx, Ty):
    songs = get_songs(dir)

    y_indexes = []
    x = []
    y = []

    for s in songs:
        x_temp, _ = graph_spectrogram(dir+s)
        y_indexes.append(get_y_index("../songs", s))
        if(x_temp.shape[1] < Tx):
            missing = Tx - x_temp.shape[1]
            zeros = np.repeat(np.zeros((x_temp.shape[0],1)), missing, axis=1) # (101,missing)
            x_temp = np.append(x_temp, zeros, axis = 1)

        x.append(x_temp)

    x = np.swapaxes(x, 1,2) # x.shape = (m, Tx, n_x)

    for i in y_indexes:

        y_oh = to_one_hot(n_y, i)
        y_ohs = set_labels(y_oh, Ty, i)
        y.append(y_ohs)
    y = np.array(y) # y.shape = (m, Ty, n_y)

    return x,y



def insert_string_in_middle(string, word):
    return string[:-4] + word + string[-4:]

def split_song(dir, no_divided_songs = 10):
    songs = get_songs(dir)
    for s in songs:
        rate, data = get_wav_info(dir+s)
        no_jumps = int(np.round(data.shape[0] / no_divided_songs))
        start = 0
        end = no_jumps
        for i in range(no_divided_songs):
            data_temp = data[start:end, :]
            start = end
            end = end + no_jumps
            name = insert_string_in_middle(s, str(i))
            wavfile.write("../songs_splited/"+name, rate, data_temp)

def balance_dimension(a,b):
    a_shape = a.shape
    b_shape = b.shape
    diff = np.abs(a_shape[0] - b_shape[0])
    if a_shape > b_shape:
        zeros = np.repeat(np.zeros((1,b_shape[1])), diff, axis=0)

        b = np.append(b,zeros,axis=0)
    elif b_shape > a_shape:
        zeros = np.repeat(np.zeros((1,a_shape[1])), diff, axis=0)

        a = np.append(a,zeros,axis=0)
    return a,b
