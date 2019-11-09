import numpy as np 

def normalize(x):

    norms = np.abs(x).sum(axis=1)
    x /= norms[:, np.newaxis]

    return x