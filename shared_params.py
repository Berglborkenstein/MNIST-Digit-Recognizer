import numpy as np

# Define weights and biases
Wi = None
bi = None
dropouti = None

def set_params(weights, biases, p):
    global Wi, bi, dropouti
    Wi = weights
    bi = biases
    dropouti = p

def get_params():
    return Wi,bi,dropouti