import numpy as np
from threading import Thread

from functions import helper_func as func, activations as act

class attention_model():
    def __init__(self, unit, A, S, n_s, layer_dimension):
        """
        Attention Model at time step t in Ty
        -----Parameter------
        S: scalar number - width of windows
        n_s: scalar number - dimension of s_prev
        layer_dimsion: dimension of hidden layer (type = list)
        A: concat hidden state of Bidirectional LSTM (Tx, 2 * n_a)
        -----Return-------
        _c_t: context at time-step t in Ty
        """
        self._layer = None
        # checking if layer_dimension argument is a list or not
        if isinstance(layer_dimension, list):
            self._layer = layer_dimension
        else:
            sys.exit("The argument of layer_dimension is not a list. Terminating operation...")

        self._A = A
        self.S = S
        self.n_a = self._A.shape[1] # n_a of concat => n_a_concat = 2 * n_a_normal
        self.n_s = n_s
        self.n_x = self.n_a + self.n_s
        self.unit = unit
        self.gradients = []
        # initialize weight for model
        # input to neural have shape = (1, n_a + n_s)
        self._params = {}
        for i in range(len(self._layer)):
            self._params["W"+str(i+1)] = func.xavier((self.n_x, self._layer[i]))
            self._params["b"+str(i+1)] = np.zeros((1, self._layer[i]))
            self.n_x = self._layer[i]

        # initialize weight for last layer
        self._params["We"] = func.xavier((self._layer[len(self._layer)-1], 1))
        self._params["be"] = np.zeros((1,1))

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

    def nn_forward_propagation(self, prev_s, start, end):
        """
        prev_s: hidden state of post-LSTM from time step t-1 in Ty (1, n_s)
        start: start index to slice self._A
        end: end index to slice self._A
        """
        _current_A = self._A[start:end, :]
        # call duplicate function fron functions module to duplicate _prev_s from (1,n_s) to (S, n_s)
        _prev_S = func.duplicate(self.S, self.n_s, prev_s, axis = 0)

        # concatenate S and A
        _SA_concat = np.concatenate((_current_A, _prev_S), axis = 1)

        # assert the dimension of concatenate (S, n_a + n_s)

        assert(_SA_concat.shape == (self.S, self.n_a + self.n_s))
        _caches_t = []
        _energies = []
        for s in range(self.S):
            energy, caches_t_s = self.nn_cell_forward(_SA_concat[s,:].reshape((1, self.n_a + self.n_s)))
            _caches_t.append(caches_t_s)
            _energies.append(energy)

        # calculate alpha
        # alphas shape = (1,S)

        _alphas = act.softmax(np.array(_energies).reshape((1,self.S)))

        # calculate context of time step t shape=(1,n_c) n_c = 2*n_a
        c = np.matmul(_alphas, _current_A)
        assert(c.shape == (1, self.unit))
        return _alphas, c, _energies, _caches_t, _current_A

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
        return d_at_s, d_s_prev_s, d_ca_s, gradients

    def nn_backward_propagation(self, dC, _alphas, _current_A, _caches_t):
        """
        ---parameters----
        dC: gradient of context (1, 2 * n_a)
        _alphas: list of alpha of attention model at time step t, each alpha have shape = (1,1)
        _current_A: list of hidden state of attention model at time step t (input) each hidden state have shape = (1, 2 * n_a)
        _caches_t: list of cache of attention model at time step t
        """
        d_AS = []
        d_s_prev = np.zeros((1, self.n_s))
        alphas = _alphas.reshape(-1)
        gradients_t = []
        for s in reversed(range(self.S)):
            alpha = np.atleast_2d(alphas[s])
            d_at_s, d_s_prev_s, d_ca_s, gradients = self.nn_cell_backward_propagation(dC, alpha, np.atleast_2d(_current_A[s]), _caches_t[s])
            d_at = d_at_s + d_ca_s
            assert(d_at.shape == (1, self.n_a))

            d_AS.append(d_at) # S -> 1

            d_s_prev = d_s_prev + d_s_prev_s
            gradients_t.append(gradients)
        d_AS = np.flip(d_AS, axis = 0) # flip 1 -> S
        return d_s_prev, d_AS, gradients_t

    # a thread function used to calc batch grads
    def gradient_thread(self, gradient_fac, result, index):
        grads = {k: np.zeros_like(v) for k,v in gradient_fac[0].items()}
        for grad in gradient_fac:
            for k in grad.keys():
                grads[k] = grads[k] + grad[k]

        result[index] = grads

    def cell_update_gradient_t(self, gradients_t, thread_no):
        """
        gradients_t: a list of dictionary of gradient at time step t
        thread_no: number of threads - scalar number
        -----return----
        None; append grads of each attention model to layer variable self.gradients
        """
        grads = {k: np.zeros_like(v) for k,v in gradients_t[0].items()}
        results = [None] * thread_no
        threads = [None] * thread_no
        # for grad in gradients_t:
        #     for k in grad.keys():
        #         grads[k] = grads[k] + grad[k]
        s = int(np.round(self.S / thread_no))
        start = 0
        end = s

        for i in range(len(threads)):
            threads[i] = Thread(target=self.gradient_thread, args=(gradients_t[start:end], results, i))
            threads[i].start()
            start = end
            end = end + s

        for i in range(len(threads)):
            threads[i].join()

        for k in grads.keys():
            for i in range(len(threads)):
                grads[k] = grads[k] + results[i][k]

        self.gradients.append(grads)

    # update layer gradient
    def update_gradient_layer(self):
        """
        ---return---
        grads: gradients of the entire layer
        """
        grads = {k: np.zeros_like(v) for k,v in self.gradients[0].items()}
        for grad in self.gradients:
            for k in grad.keys():
                grads[k] = grads[k] + grad[k]

        return grads

    def update_layer(self, lr=0.001):
        """
        ----parameters-----
        lr: learning rate
        """
        grads = self.update_gradient_layer()
        for i in range(len(self._layer)):
            self._params["W"+str(i+1)] = self._params["W"+str(i+1)] - lr*grads["dW"+str(i+1)]
            self._params["b"+str(i+1)] = self._params["b"+str(i+1)] - lr*grads["db"+str(i+1)]

        self._params["We"] = self._params["We"] - lr*grads["dWe"]
        self._params["be"] = self._params["be"] - lr*grads["dbe"]
