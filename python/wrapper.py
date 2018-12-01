import numpy as np
import copy


class Bidirectional():
    def __init__(self, name, layer):
        self.name = name
        self.forward = copy.copy(layer)
        self.forward.name = "biDirectional_" + self.forward.name + "_forward"
        self.backward = copy.copy(layer)
        self.backward.is_backward = True
        self.backward.name = "biDirectional_"+ self.backward.name +"_backward"


        self.dA_forward = None
        self.dA_backward = None

        self.concat = None
    def bi_forward(self, X):
        """
        -----------------------
        Return:
            A_forward: forward hidden state of all time-step (Tx, n_a) -->
            A_backward: backward hidden state of all time-step (Tx, n_a) <--
        """
        print("Calculating A forward.....")
        A_forward = self.forward.forward_propagation(X)
        print("Calculating A backward.....")
        A_backward = self.backward.forward_propagation(X)
        return A_forward, A_backward


    # concat forward hidden state and backward hidden state
    def concatLSTM(self, X):
        A_forward, A_backward = self.bi_forward(X)
        self.concat = np.concatenate((A_forward, A_backward), axis = 1) # shape = (Tx, 2*n_a)
        return self.concat

    def accumulate_dA(self, att_dA_list, jump_step, Ty, Tx):
        # att_dA_list 1 -> Ty
        # take first dA of first list to get n_a
        n_a = att_dA_list[0][0].shape[1]
        S = len(att_dA_list[0])
        # initialize shape of dA --- dA.shape == A.shape
        dA = np.zeros((Tx, n_a))
        start = 0
        end = S
        for att_dA in att_dA_list:
            dA[start:end,:] = dA[start:end,:] + np.array(att_dA.reshape((S, n_a)))
            start = start + jump_step
            end = end + jump_step
        self.dA_forward = dA[:, :int(n_a/2)]
        self.dA_backward = dA[:, int(n_a/2):]

    def cell_backpropagation(self, att_dA_list, jump_step, Ty, Tx):

        self.accumulate_dA(att_dA_list, jump_step, Ty, Tx)
        _ = self.forward.backward_propagation(self.dA_forward)
        _ = self.backward.backward_propagation(self.dA_backward)
        del _
    def update_weight(self, lr, i):
        self.forward.update_weight(lr, i)
        self.backward.update_weight(lr, i)
