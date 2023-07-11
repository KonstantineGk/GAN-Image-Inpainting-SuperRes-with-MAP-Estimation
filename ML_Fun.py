import numpy as np

# ReLu
def relu(x):
    return (np.maximum(0,x))

# ReLU derivative
def relu_der_(x):
    if x <= 0: y = 0
    elif x > 0: y = 1
    return y
relu_der = np.vectorize(relu_der_)

# Sigmoid
def sigmoid(x):
    return 1 / ( 1 + np.exp(x) )

# Sigmoid derivative
def sigmoid_der(x):
    return - np.exp(x) / ( (1 + np.exp(x)) ** 2 )