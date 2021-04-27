
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, exp, pi
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double gauss_1d(double[:,::1] data, double std):
    cdef double cent, g_x, x, ex, g_e
    cdef int i
    cdef list g = []

    cent = len(data) / 2
    g_x = 1 / (pi * 2 * std)
    for i in range(len(data)):
        x = cent - i
        ex = x ** 2 / (2 * std ** 2)
        g_e = exp(-ex)
        g.append(g_x * g_e)
    return g

#pointer to average style
ctypedef double (*average_)(double[:,::1], int,int,int)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double gauss_2d(double[:,::1] data , int xdim , int ydim, int std):
    cdef int i, j
    cdef double x, y, ex, ge, w , cent_x, cent_y

    cent_x = xdim / 2
    cent_y = ydim / 2
    w = 0
    for i in range(xdim):
        for j in range(ydim):
            x = cent_x - i
            y = cent_y - j
            if std == 0:
                ex = 0
            else:
                ex = (x ** 2 + y ** 2) / (2 * std ** 2)
            g_e = exp(-ex)
            w += data[i, j] * g_e

    return w / (xdim * ydim)


@cython.boundscheck(False)
@cython.wraparound(False)
def gauss_OBP(double[:,::1] data, int xdim, int ydim, int std, int final_size):
    ##assuming data is square
    cdef average_ gauss
    cdef int i, j, in_box, num_inc, box_std
    cdef list ave_final = []
    cdef list ave_general = []
    cdef tuple where, where_x, where_y
    # cdef long [:,::1] box
    cdef double [:,::1] box
    # cdef np.ndarray[dtype = long, ndim = 1] corners, point_location
    cdef np.ndarray[dtype = double, ndim = 1] corners, point_location
    cdef double ave
    cdef double [:] data_one_dim
    # cdef np.ndarray[dtype = double, ndim = 2] ave_final = np.empty(shape = (final_size, final_size))

    gauss = &gauss_2d
    in_box = (xdim/final_size)
    if std == 0:
        num_inc = 0
    else:
        num_inc = ((3 * std )/in_box)

    box_std = in_box + num_inc
    if ydim < 1:
        # data_one_dim = data[:,1]
        # print(data[:,:].shape)
        """1 dimensional case box is row"""
        for i in range(0, xdim, in_box):
            corners = np.asarray([i-box_std, i+box_std], dtype = 'double')
            if np.where(corners < 0):
                    w = np.where(corners < 0)
                    corners[w] = 0
            if np.where(corners > xdim):
                where_x = np.where(corners > xdim)
                corners[where_x] = xdim
            box = data[int(corners[0]):int(corners[1]), 0:1]
            ave = gauss(box, box.shape[0], box.shape[1], std)
            ave_final.append(ave)

    if xdim & ydim > 1:
        for i in range(0, xdim, in_box):
            for j in range(0, ydim, in_box):
                corners = np.asarray([i-box_std, i+box_std, j-box_std, j+box_std], dtype = 'double')
                # corners = corners.astype('double')
                # point_location = np.asarray([corners[1] - box_std, corners[3] - box_std])
                if np.where(corners < 0):
                    w = np.where(corners < 0)
                    corners[w] = 0
                if np.where(corners > xdim) or np.where(corners > ydim):
                    where_x = np.where(corners > xdim)
                    where_y = np.where(corners > ydim)
                    corners[where_x] = xdim
                    corners[where_y] = ydim
                # print(data[int(corners[0]), int(corners[1])])
                box = data[int(corners[0]):int(corners[1]), int(corners[2]):int(corners[3])]

                # ave_general.append(np.mean(box))
                ave = gauss(box, box.shape[0], box.shape[1], std)
                ave_final.append(ave)

    return ave_final


























