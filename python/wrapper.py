import numpy as np
import copy


class Bidirectional():
    def __init__(self, layer, a0, X):
        self.forward = copy.copy(layer)

        self.backward = copy.copy(layer)
        self.backward.is_backward = True


        self.a0 = a0
        self.X = X

    def bi_forward(self):
        caches_forward = self.forward.LSTM_layer_forward(self.a0, self.X)
        caches_backward = self.backward.LSTM_layer_forward(self.a0, self.X)
        return caches_forward, caches_backward


    # concat forward hidden state and backward hidden state
    def concatLSTM(self):
        caches_forward, caches_backward = self.bi_forward()
        cache = (caches_forward, caches_backward)
        self.A = []

        for c, cb in zip(caches_forward, caches_backward):
            _, a_forward = c
            _, a_backward = cb
            concat = np.concatenate((a_forward, a_backward), axis = 1)

            self.A.append(concat.reshape(-1)) # (Tx, 2*n_a)
        return np.array(self.A)
