import numpy as np
cimport numpy as np
from libc.math cimport sqrt, sin, exp, cos, pi
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def wave_simp(double[:,::1] grid,int xdim, int  ydim ):
    cdef np.ndarray[dtype = double, ndim = 2] wave = np.empty(shape=(xdim, ydim))
    cdef double W
    cdef int i, j

    for i in range(xdim):
        for j in range(ydim):
            wave[i, j] = 5 * cos((6.28 / 40) * i - (6.28 / 100) * 0.4 + 0) \
                     + 5 * sin((6.28 / 45) * j - (6.28 / 600) * 0.36 + 0)

    return wave


ctypedef double (*observ_ptr)(double[:,::1], int, int, int, int)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double ptr_obs(double[:,::1] SSH, int p0,  int p1, int xdim, int ydim) :
# cdef ptr_obs(np.ndarray[dtype = double, ndim = 2] SSH,
#              int p0,  int p1, int xdim, int ydim):
    cdef double obs, ssh_weight, distance, ptr, wave
    cdef int i,j
    obs = 0
    for i in range(xdim):
        for j in range(ydim):
            distance = sqrt((p0 - i) ** 2 + (p1 - j) ** 2)
            d_d = distance/0.75
            if distance == 0:
                ptr = 1/ 0.75
            else:
                ptr= ((sin(pi*d_d) ** 2 )/ ((pi*d_d) ** 2))/ 0.75
            ssh_weight = SSH[i, j]* ptr
            obs += ssh_weight
    return obs


@cython.boundscheck(False)
@cython.wraparound(False)
def ptr_store(int xdim, int ydim, double[:,::1] SSH ):
# def ptr_store(int xdim, int ydim, np.ndarray[dtype = double, ndim = 2] SSH):
    cdef observ_ptr obs_func
    cdef np.ndarray[dtype = double, ndim = 2] SSH_observed
    cdef int i, j
    cdef double observed

    obs_func = &ptr_obs
    SSH_observed = np.empty_like(SSH)
    for i in range(xdim):
        for j in range(ydim):
            observed = ptr_obs(SSH, i, j, xdim, ydim )
            SSH_observed[i,j] = observed
    return SSH_observed


























