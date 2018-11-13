import numpy as np
from functions import helper_func as func

class attention_model():
    def __init__(self, _current_A, _prev_s, layer_dimension):
        """
        Attention Model at time step t in Ty
        -----Parameter------
        _current_A: hidden state of bi-Directional LSTM from time step 1 to S... 2 to S + 1.... (sliding windows) (S, n_a)
        _prev_s: hidden state of post-LSTM from time step t-1 in Ty (1, n_s)
        layer_dimsion: dimension of hidden layer (type = list)
        -----Return-------
        _c_t: context at time-step t in Ty
        """
        self._layer = None
        # checking if layer_dimension argument is a list or not
        if isinstance(layer_dimension, list):
            self._layer = layer_dimension
        else:
            sys.exit("The argument of layer_dimension is not a list. Terminating operation...")

        self._current_A = _current_A
        self._prev_s = _prev_s
        self.S = _current_A.shape[0]
        self.n_s = _prev_s.shape[1]

        # initialize weight for model
        self._params = []
        for i in range(len(self._layer)):
            self._parms["W"+str(i+1)] = func.xavier(())

        # call duplicate function fron functions module to duplicate _prev_s from (1,n_s) to (S, n_s)
        self._prev_S = func.duplicate(self.S, self.n_s, self._prev_s, axis = 0)

        # concatenate S and A
        self._SA_concat = np.concatenate((self._current_A, self._prev_S), axis = 1)

        # assert the dimension of concatenate (S, n_a + n_s)
        assert(self._SA_concat.shape == (self.S, self._current_A.shape[1] + self.n_s))

    def cell_forward(self):
        # hidden layer
        for h in self._layer: