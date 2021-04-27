import numpy as np
import matplotlib.pyplot as plt
import glob as glob
from natsort import natsorted


def create_grid(xdim, ydim):
    return np.empty(shape=(xdim, ydim))


def std_dev(X):
    """
    :param X:sample where observations are given in rows
    :return: variance and standard dev
    """
    av_ = np.mean(X)
    print(av_)
    s = 0
    for i in range(len(X)):
        s += (X[i] - av_) ** 2
        s = s / (len(X) - 1)
    return s, np.sqrt(s)


def cov(X, Y, xdim, ydim):
    a = np.empty(shape=(xdim, ydim))
    x_bar = np.nanmean(X)
    y_bar = np.nanmean(Y)

    for i in range(xdim):
        for j in range(ydim):
            a[i, j] = (1 / xdim - 1) * np.sum((X[i] - x_bar) * (Y[j] - y_bar))
    return a


# a = cov(samp_2, samp_, 10,10)


def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA ** 2).sum(1)
    ssB = (B_mB ** 2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))


def corr_coef(A, B):
    A_ = A - A.mean()
    B_ = B - B.mean()

    ssA = (A_ ** 2).sum()
    ssB = (B_ ** 2).sum()
    fin = np.dot(A_, B_.T) / np.sqrt(np.dot(ssA, ssB))
    return fin


if __name__ == '__main__':
    data1 = np.loadtxt('/Users/caggiano/SWOT_GIT/waves/GIT/gauss/output/std_0').flatten()
    data_list = natsorted(glob.glob('/GIT/gauss/output/*'))
    cor_list = []
    for i in range(len(data_list)):
        data2 = np.loadtxt(data_list[i]).flatten()
        cor = corr_coef(data1, data2)
        cor_list.append(cor)
    line_0 = np.linspace(0, 18, 18)
    line_00 = np.empty(shape=18)
    line_00[:] = 0
    plt.plot(cor_list, 'o', color='blue')
    plt.plot(cor_list)
    plt.plot(line_0, line_00, color='grey', linestyle='--')
    plt.xlabel('Standard deviation (km)')
    plt.ylabel('Correlation')
    plt.show()
