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
        self._alphas = None
        self._caches_t = []
        # call duplicate function fron functions module to duplicate _prev_s from (1,n_s) to (S, n_s)
        self._prev_S = func.duplicate(self.S, self.n_s, self._prev_s, axis = 0)

        # concatenate S and A
        self._SA_concat = np.concatenate((self._current_A, self._prev_S), axis = 1)
        n_x = self._SA_concat.shape[1]

        # initialize weight for model
        # input to neural have shape = (1, n_a + n_s)
        self._params = {}
        for i in range(len(self._layer)):
            self._params["W"+str(i+1)] = func.xavier((n_x, self._layer[i]))
            self._params["b"+str(i+1)] = np.zeros((1, self._layer[i]))
            n_x = self._layer[i]

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
        cache_last = (input, Z_last, energy)
        caches_t_s.append(cache_last)

        return energy, caches_t_s

    def nn_forward_propagation(self):
        _energies = []
        for s in range(self.S):
            energy, caches_t_s = self.nn_cell_forward(self._SA_concat[s,:].reshape((1, self.n_a + self.n_s)))
            self._caches_t.append(caches_t_s)
            _energies.append(energy)

        # calculate alpha
        # alphas shape = (1,S)

        self._alphas = act.softmax(np.array(_energies).reshape((1,self.S)))

        # calculate context of time step t shape=(1,n_c) n_c = 2*n_a
        c = np.matmul(self._alphas, self._current_A)

        return self._alphas, c, _energies, self._caches_t

    def nn_cell_backward_propagation(self, dC, alpha, a_s, cache_t_s):
        """
        ---return----
            d_at_s: gradient of hidden state from alpha
            d_ac_s: gradient of hidden state from context
            d_s_prev_s: gradient of hidden state of post_LSTM
        """
        gradients = {}
        # dC shape = (1, 2*n_a)
        # a_s = (1, 2 * n_a)
        # shape = (1, 1)
        d_alpha = np.matmul(dC, np.transpose(a_s))

        # shape = (1, 1)
        d_energy = d_alpha * (alpha * (1 - alpha))
        first = True
        # not count last layer
        count = len(cache_t_s) - 1
        d_Z_last = None
        W = self._params["We"]

        for cache in reversed(cache_t_s):
            input, Z, e = cache
            if (first):
                # shape = (1 , 1)
                d_Z_last = d_energy * act.backward_relu(Z)

                # shape = (1,10)
                dWe = np.matmul(d_Z_last, input)
                dbe = d_Z_last
                gradients["dWe"] = dWe
                gradients["dbe"] = dbe
                first = False
            else:
                dZ = np.matmul(d_Z_last, np.transpose(W)) * act.backward_tanh(Z)
                dW = np.matmul(np.transpose(input), dZ)
                db = dZ
                gradients["dW"+str(count)] = dW
                gradients["db"+str(count)] = db
                W = self._params["W"+str(count)]
                d_Z_last = dZ
                count = count - 1
        d_input = np.matmul(d_Z_last, np.transpose(W)) # shape = (1, 2 * n_a + n_s)
        d_at_s = d_input[:, :self.n_a] # shape = (1, 2 * n_a)
        d_s_prev_s = d_input[:, self.n_a:] # shape = (1, n_s)
        d_ca_s = dC * alpha
        return d_at_s, d_s_prev_s, d_ca_s

    def nn_backward_propagation(self, dC):
        d_AS = []
        d_s_prev = np.zeros((1, self.n_s))
        for s in reversed(range(self.S)):
            d_at_s, d_s_prev_s, d_ca_s = nn_cell_backward_propagation(dC, np.atleast_2d(self._alphas[s]), np.atleast_2d(self._current_A[s]), self._caches_t[s])
            d_AS.append(d_at_s + d_ca_s)
            d_s_prev = d_s_prev + d_s_prev_s

        return d_s_prev, d_AS
