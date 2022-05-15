import numpy as np

def Dataloader(X, y, size, batch_size):
    idx = np.random.randint(low=0, high=(size-1), size=batch_size)
    return X[idx], y[idx]