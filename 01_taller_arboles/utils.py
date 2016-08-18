import numpy as np

def log2(v):
    return np.log(v)/np.log(2)

def frequencies(y):
    counts = {}

    for elem in y:
        if elem in counts:
            counts[elem] += 1
        else:
            counts[elem] = 1

    return counts.keys(), np.array(counts.values(), dtype=float) / len(y)

def entropy(y):
    labels, freqs = frequencies(y)

    return -sum(freqs * log2(freqs))

def information_gain(column,y):
    # Completar
    pass
