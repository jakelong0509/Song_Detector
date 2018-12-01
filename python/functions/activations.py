import numpy as np

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def hard_sigmoid(Z):
    y = 0.2 * Z + 0.5
    return np.clip(y, 0, 1)


def relu(Z):
    return (Z > 0) * Z

def softmax(Z):
    e_x = np.exp(Z - np.max(Z))
    return e_x / e_x.sum(axis = 1)

def backward_sigmoid(a):
    return a * (1 - a)

def backward_tanh(Z):
    return 1-(np.tanh(Z))**2

def backward_relu(Z):
    mul = np.ones(Z.shape)
    return (Z > 0) * mul

def backward_softmax(t_hat, i):
    t_hat[i] = t_hat[i] - 1
    return t_hat

# based on keras drop out
def dropout(input, level):
    noise_shape = input.shape # (Tx, n_a)
    noise = np.random.choice([0,1], noise_shape, replace = True, p=[level, 1-level])
    return input * noise / (1 - level), noise / (1 - level)
