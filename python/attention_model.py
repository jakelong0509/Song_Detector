import numpy as np
from functions import helper_func as func, activations as act

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
        self.S, self.n_a = _current_A.shape
        self.n_s = _prev_s.shape[1]


        # call duplicate function fron functions module to duplicate _prev_s from (1,n_s) to (S, n_s)
        self._prev_S = func.duplicate(self.S, self.n_s, self._prev_s, axis = 0)

        # concatenate S and A
        self._SA_concat = np.concatenate((self._current_A, self._prev_S), axis = 1)
        n_x = self._SA_concat.shape[1]

        # initialize weight for model
        # input to neural have shape = (1, n_a + n_s)
        self._params = []
        for i in range(len(self._layer)):
            self._params["W"+str(i+1)] = func.xavier((n_x, self._layer[i]))
            self._params["b"+str(i+1)] = np.zeros((1, self._layer[i]))
            n_x = self.layer[i]

        # initialize weight for last layer
        self._params["We"] = func.xavier((self._layer[len(self._layer)-1], 1))
        self._params["be"] = np.zeros((1,1))

        # assert the dimension of concatenate (S, n_a + n_s)
        assert(self._SA_concat.shape == (self.S, self._current_A.shape[1] + self.n_s))

    def nn_cell_forward(self, curr_input):
        # hidden layer
        assert(curr_input.shape == (1, self.n_a + self.n_s))
        input = curr_input # shape = (1, n_a + n_s)
        caches_t_s = []
        for i in range(len(self._layer)):
            Z = np.matmul(input, self._params["W"+str(i+1)]) + self._params["b"+str(i+1)]
            e = act.tanh(Z)
            cache = (input, Z, e)
            caches_t_s.append(cache)
            input = e

        # last layer
        Z_last = np.matmul(input, self._params["We"]) + self._params["be"]
        energy = act.relu(Z_last)
        cache_last = (input, Z, e)
        caches_t_s.append(cache_last)

        return energy, caches_t_s

    def nn_forward_propagation(self):
        _caches_t = []
        _energies = []
        for s in range(self.S):
            energy, caches_t_s = cell_forward(self._SA_concat[s,:])
            _caches_t.append(caches_t_s)
            _energies.append(energy)

        # calculate alpha
        # alphas shape = (1,S)
        alphas = act.softmax(np.Array(_energies.reshape((1,self.S))))

        # calculate context
        c = np.matmul(alphas, self._current_A)
        return alphas, c, _energies, _caches_t
