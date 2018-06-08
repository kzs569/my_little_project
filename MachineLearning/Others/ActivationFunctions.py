import numpy as np
import pandas as pd
import math


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def tanh(x):
    return 2 * sigmoid(2 * x) - 1


def relu(x):
    return x if x > 0 else 0


def leaky_relu(x, a=0.1):
    return x if x > 0 else a * x


def softplus(x):
    return np.log(1 + np.exp(x))


def erelu(x, a=0.1):
    return x if x > 0 else a * (np.exp(x) - 1)
