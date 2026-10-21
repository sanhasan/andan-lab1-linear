import numpy as np


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def load_dataset(path):
    with np.load(path) as data:
        return data["X"], data["y"]
