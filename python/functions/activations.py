import numpy as np

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def hard_sigmoid(Z):
    y = 0.2 * Z + 0.5
    return np.clip(y, 0, 1)

def tanh(Z):
    return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

def relu(Z):
    return np.atleast_2d(max(0,Z))

def softmax(Z):
    deli = np.sum(np.exp(Z))
    return np.exp(Z) / deli

def backward_sigmoid(a):
    return a * (1 - a)

def backward_tanh(Z):
    return 1-(tanh(Z))**2

def backward_relu(Z):
    if Z > 0:
        return 1
    elif Z <= 0:
        return 0

def backward_softmax(t_hat, i):
    t_hat[i] = t_hat[i] - 1
    return t_hat

# based on keras drop out
def dropout(input, level):
    noise_shape = input.shape # (Tx, n_a)
    noise = np.random.choice([0,1], noise_shape, replace = True, p=[level, 1-level])
    return input * noise / (1 - level), noise / (1 - level)
