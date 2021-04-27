import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import ifft, fft

# wave = np.loadtxt('/Users/caggiano/SWOT_GIT/waves/GIT/wave_gen/real_50k_wave')
# ssh =  = np.loadtxt('/Users/caggiano/SWOT_GIT/waves/GIT/ptr/wave_form_noise_50km')
# sea_surf_height = ssh #+ wave
# ptr_ = np.loadtxt('/Users/caggiano/SWOT_GIT/waves/GIT/ptr/ptr_ssh_inc_wave_50km')

#gauss needs it


def gauss_1d(data, xdim, std, final_len):
    in_box = int(xdim / final_len)

    if std == 0:
        num_inc = 0
    else:
        num_inc = (3 * std) / in_box
    box_std = int(in_box + num_inc)
    listt = []
    for i in range(0, xdim, in_box):
        corners = np.asarray([i - box_std, i + box_std])
        if np.where(corners < 0):
            w = np.where(corners < 0)
            corners[w] = 0
        if np.where(corners > xdim):
            w = np.where(corners > xdim)
            corners[w] = xdim
        box = data[corners[0]:corners[1]]
        listt.append(np.mean(box))
        # center = box.size/2
        # w = 0
        # for j in range(box.size):
        #     x = center - j
        #     print(x)
        #     if std == 0:
        #         ex = 0
        #     else:
        #         ex = (x**2)/(2*std**2)
        #     w+= box[j] * np.exp(-ex)
        # listt.append(w)
    return np.asarray(listt)


def corr_coef(A, B):
    A_ = A - A.mean()
    B_ = B - B.mean()

    ssA = (A_ ** 2).sum()
    ssB = (B_ ** 2).sum()
    fin = np.dot(A_, B_.T) / np.sqrt(np.dot(ssA, ssB))
    return fin


list_cor = []
list_gauss = []
# ave_0 = gauss_1d(ptr_, 50000, 0, 1000)
ave_0 = sea_surf_height
for i in range(0, 5000, 1000):
    ave = gauss_1d(ptr_, 50000, i, 1000)
    list_gauss.append(ave)
    list_cor.append(corr_coef(ave_0[0:1000], ave))

plt.plot(list_cor, marker='o')
plt.plot(np.arange(0, 5), np.tile(0, (5)), color='grey',
         linestyle='-.')
plt.xlabel('Seperation (km)')
plt.ylabel('Correlation')
plt.show()

plt.plot(sea_surf_height[0:1000], 'red')
plt.plot(list_gauss[4], 'b')
plt.show()






