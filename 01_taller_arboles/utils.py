import numpy as np
from collections import Counter

def log2(v):
    return np.log(v)/np.log(2)

def entropy(y):
    counter = Counter(y)

    freqs = np.array(counter.values(), dtype=float) / len(y)

    return -sum(freqs * log2(freqs))

def information_gain(column,y):
    # Completar
    pass
