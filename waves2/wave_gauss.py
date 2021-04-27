import numpy as np
import matplotlib.pyplot as plt


def gauss_1d(data, std):
    g = []
    cent = len(data) / 2
    g_x = 1 / (np.pi * 2 * std)
    for i in range(len(data)):
        x = cent - i

        ex = x ** 2 / (2 * std ** 2)

        g_e = np.exp(-ex)
        g.append(g_x * g_e)
    return g


def gauss_2d(data, xdim, ydim, std):
    cent_x = xdim / 2
    cent_y = xdim / 2
    w = 0
    for i in range(xdim):
        for j in range(ydim):
            x = cent_x - i
            y = cent_y - j
            ex = (x ** 2 + y ** 2) / (2 * std ** 2)
            g_e = np.exp(-ex)
            w += data[i, j] * g_e

    return w / (xdim * ydim)


def gauss_box(data, xdim, ydim, std, box):
    """
    :param data:
    :param xdim:
    :param ydim:
    :param std:
    :param box:
    :return: Gaussian smoothed data, no reduction
    """
    w = []
    for i in range(xdim):
        print(i)
        for j in range(ydim):
            corners = np.asarray([i - box, i + box, j - box, j + box])

            if np.where(corners < 0):
                where = np.where(corners < 0)
                corners[where] = 0
            if np.where(corners > xdim) or np.where(corners > ydim):
                where_x = np.where(corners > xdim)
                where_y = np.where(corners > ydim)
                corners[where_x] = xdim
                corners[where_y] = ydim
            boxed = data[corners[0]:corners[1], corners[2]:corners[3]]
            box_x = boxed.shape[0]
            box_y = boxed.shape[1]
            w.append(gauss_2d(boxed, box_x, box_y, std))
    return np.asarray(w).reshape(xdim, ydim)


def gauss_box_reduce(data, xdim, ydim, box, std):
    """
    :param data: full data
    :param xdim: dim of data
    :param ydim: dim of data
    :param box: box size
    :param std: std
    :return: reduced weighted data, more box car
    """
    w = []
    box_inx = int(xdim / box)
    box_iny = int(ydim / box)

    for i in range(0, xdim, box):
        print(i)
        for j in range(0, ydim, box):
            corners = np.asarray([i - box, i + box, j - box, j + box])

            if np.where(corners < 0):
                where = np.where(corners < 0)
                corners[where] = 0
            if np.where(corners > xdim) or np.where(corners > ydim):
                where_x = np.where(corners > xdim)
                where_y = np.where(corners > ydim)
                corners[where_x] = xdim
                corners[where_y] = ydim
            boxed = data[corners[0]:corners[1], corners[2]:corners[3]]
            box_x = boxed.shape[0]
            box_y = boxed.shape[1]
            w.append(gauss_2d(boxed, box_x, box_y, std))
    return np.asarray(w).reshape(box_inx, box_iny)


# def gauss_OBP(data, xdim, ydim, std, final_size):
#     """
#     :param final_size: final data size for box size
#     :param data: full data
#     :param xdim: data dim
#     :param ydim: data dim
#     :param std: standard dev in meters
#     :return: full reduction with looks at bins next to 3*std
#     """
#
#     in_boxx = int(xdim / final_size)  ##number of things in box (size/distance)
#     in_boxy = int(ydim / final_size)  ##number of things in box
#
#     first_split = np.array_split(data, final_size)  ##list
#     # part = []
#     # for i in range(len(first_split)):
#     #     part.append(np.hsplit(first_split[i], final_size))
#     # part_ = np.asarray(part).reshape(final_size * final_size, in_boxx, in_boxy)
#     #partitions made, now gather for larger look squares
#     #squares included need to cover 3*std in both x and y direction
#     num_includ_y = (3 * std)/in_boxx     ## meters change to boxes
#     num_includ_y = (3 * std)/in_boxy
#     # box_inc = num_includ_y/part_[0].shape[0]
#     # index = np.arange(part_.shape[0]).reshape(final_size, final_size)
#     ##gather num_include from part_
#
#     return num_includ_y


def gauss_OBP(data, xdim, ydim, std, final_size):
    in_boxx = int(xdim / final_size)  ##number of things in box (size/distance)
    in_boxy = int(ydim / final_size)  ##number of things in box
    num_inc = int((3 * std) / in_boxx)
    in_ = in_boxx + num_inc
    # in_x = in_boxx + num_inc
    # in_y = in_boxy + num_inc
    c = []
    ave = []
    for i in range(1, data.shape[0], in_boxx):  ##forward size of box
        print(i,'/',data.shape[0])
        for j in range(1, data.shape[1], in_boxy):
            up_down = np.array([i - in_, i + in_])
            side_side = np.array([j - in_, j + in_])
            up_down[np.where(up_down < 0)] = 0
            side_side[np.where(side_side < 0)] = 0
            center = data[i - in_:i + in_, j - in_:j + in_]
            c.append(center)
            if center.size > 0:
                ave.append(gauss_2d(center, center.shape[0], center.shape[1], std))

    return np.asarray(ave).reshape(int(np.sqrt(len(ave))), int(np.sqrt(len(ave))))
