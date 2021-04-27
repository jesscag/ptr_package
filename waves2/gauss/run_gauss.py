import numpy as np
from guass_c import gauss_OBP
import matplotlib.pyplot as plt
import time

def corr_coef(A, B):
    A_ = A - A.mean()
    B_ = B - B.mean()

    ssA = (A_ ** 2).sum()
    ssB = (B_ ** 2).sum()
    fin = np.dot(A_, B_.T) / np.sqrt(np.dot(ssA, ssB))
    return fin

# if __name__ == "__main__":
#     # import data
#     t0 = time.time()
#     # data = data = np.loadtxt('/Users/caggiano/SWOT_GIT/waves/GIT/ptr/obs5x5')
#     data = np.loadtxt('/Users/caggiano/SWOT_GIT/waves/GIT/ptr/obs50_56s')
#     data = np.ascontiguousarray(data)
#     # data = np.tile(data, (10, 10))
#
#     original = np.loadtxt('/Users/caggiano/SWOT_GIT/waves/GIT/interp_hycom_56s')
#
#     ave = []
#     for i in range(0,6000, 1000):
#         # print(i)
#         std = i
#
#         # final_size = 1000
#
#         ## tile data for larger swath (50 km x 50 km)
#         xdim = data.shape[0]
#         ydim = data.shape[1]
#         # t1 = time.time()
#         ave.append(gauss_OBP(data, xdim, 0, std, 1000))
#         # t2 = time.time()
#         # print(t2 - t1)
#         # print('total elapsed ', t2 - t0)
#         ## turn into array for plot
#         # shape = int(np.sqrt(len(ave_weight)))
#         # smoothed_array = np.asarray(ave_weight).reshape(1, 1000)
#         # np.savetxt('/Users/caggiano/SWOT_GIT/waves/GIT/gauss/output/std_'+str(std), smoothed_array)
#     # for i in range(len(ave)):
#     #     coef = corr_coef(original[0:1000], np.asarray(ave[i]))
#     #     print(coef)
#
#
#     for i in range(0,len(ave)):
#         plt.plot(ave[i], label = str(i))
#     plt.plot(original[0:1000], 'o')
#     plt.legend()
#     plt.show()
