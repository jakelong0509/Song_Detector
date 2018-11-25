import numpy as np
import copy


class Bidirectional():
    def __init__(self, layer, X):
        self.forward = copy.copy(layer)

        self.backward = copy.copy(layer)
        self.backward.is_backward = True



        self.X = X

    def bi_forward(self):
        """
        -----------------------
        Return:
            A_forward: forward hidden state of all time-step (Tx, n_a) -->
            A_backward: backward hidden state of all time-step (Tx, n_a) <--
        """
        print("Calculating A forward.....")
        A_forward = self.forward.forward_propagation(self.X)
        print("Calculating A backward.....")
        A_backward = self.backward.forward_propagation(self.X)
        return A_forward, A_backward


    # concat forward hidden state and backward hidden state
    def concatLSTM(self):
        A_forward, A_backward = self.bi_forward()
        return np.concatenate((A_forward, A_backward), axis = 1) # shape = (Tx, 2*n_a)
