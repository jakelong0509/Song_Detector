import numpy as np
import copy


class Bidirectional():
    def __init__(self, layer, X):
        self.forward = copy.copy(layer)

        self.backward = copy.copy(layer)
        self.backward.is_backward = True


        self.dA_forward = None
        self.dA_backward = None
        self.X = X
        self.Tx, self.n_x = X.shape
        self.concat = None
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
        self.concat = np.concatenate((A_forward, A_backward), axis = 1) # shape = (Tx, 2*n_a)
        return self.concat

    def accumulate_dA(self, att_dA_list, jump_step, Ty):
        # att_dA_list 1 -> Ty
        # take first dA of first list to get n_a
        n_a = att_dA_list[0][0].shape[1]

        # initialize shape of dA --- dA.shape == A.shape
        dA = np.zeros((self.Tx, n_a))

        start = 0
        end = S

        for att_dA in att_dA_list:
            dA[start:end,:] = dA[start:end,:] + np.array(att_dA.reshape((S, n_a)))
            start = start + jump_step
            end = end + jump_step

        self.dA_forward = dA[:, :n_a]
        self.dA_backward = dA[:, n_a:]

    def cell_backpropagation(self, att_dA_list, jump_step, Ty):
        
        self.accumulate_dA(att_dA_list, jump_step, Ty)

        forward_gradients = self.forward.backward_propagation(self.dA_forward)
        backward_gradients = self.backward.backward_propagation(self.dA_backward)

        self.forward.update_weight(forward_gradients, lr = 0.005)
        self.backward.update_weight(forward_gradients, lr = 0.005)
