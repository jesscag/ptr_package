from .ptr import ptr_store
import numpy as np

def ptr_run(SSH):
    x_for = 2
    xdim = SSH.shape[0]
    ydim = SSH.shape[1]
    SSH = np.ascontiguousarray(SSH)

    o = []
    for i in range(0,xdim, x_for):
        observed = ptr_store(x_for, ydim, SSH[i:i+x_for])
        o.append(observed)
    obs = np.asarray(o).reshape(xdim,ydim)
    return obs