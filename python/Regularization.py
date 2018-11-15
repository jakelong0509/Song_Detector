import numpy as np
import os
import sys

class regularization():
    def __init__(self, hidden_state):
        # shape of hidden_state = (Tx, 2 * n_a)
        self.hidden_state = hidden_state

    def dropout(self, ):
        shape = self.hidden_state.shape
        eye = np.eye(shape)
