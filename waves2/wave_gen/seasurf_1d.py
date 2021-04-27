import numpy as np
from spectra import spectra_PM, spectra_ECKV
from cmath import sqrt
from scipy.fft import ifft, fft
import matplotlib.pyplot as plt


def compute_amplitudes(L, nsamp):
    """
    :param L:
    :param nsamp:
    :return:
    """
    # determine kfund/nyquist
    delta_x = L / nsamp
    k_fund = np.pi * 2 / L
    k_nyq = (nsamp / 2) * k_fund
    nk_pos = nsamp / 2

    delta_k = (k_nyq - k_fund) / (nk_pos - 1)
    kpos = np.arange(delta_k, k_nyq + k_fund, delta_k)
    k = np.hstack([0, kpos, -kpos[::-1][1:]])
    SK = spectra_PM(kpos, 15)
    two_sided = np.hstack([0, SK, -SK[::-1][1:]])
    S2root = []
    for i in range(len(two_sided)):
        S2root.append(sqrt(two_sided[i] * delta_k))
    S2root = np.asarray(S2root)

    C3 = 1 / np.sqrt(8)
    rj = np.random.normal(size=nsamp)
    sj = np.random.normal(size=nsamp)
    z_hat = np.empty(shape=nsamp, dtype='complex128')
    z_hat[0] = complex(0, 0)
    for j in range(1, nsamp):
        z_hat[j] = C3 * S2root[j] * complex(rj[j] + rj[nsamp - j], sj[j] - sj[nsamp - j])
        # z_hat[j] = C3 * S2root[j] * complex(rj[j], sj[j])
    return z_hat*N, k, two_sided, SK, delta_k


def compute_periodogram(ifft_ssh, nsamp, delta_k):
    comp_z = []
    for i in range(0, len(ifft_ssh)):
        comp_z.append(complex(ifft_ssh[i]))
    zhat_amp = fft(comp_z)/nsamp
    P1S = np.empty(shape=int(nsamp / 2) + 1)
    P1S[:] = np.nan
    P1S[0] = np.abs(zhat_amp[0] ** 2)
    P1S[int(nsamp / 2)] = np.abs(zhat_amp[int(nsamp / 2)]) ** 2
    for j in range(0, int(nsamp / 2)):
        P1S[j] = 2 * np.abs(zhat_amp[j]) ** 2
    P1S = P1S / delta_k
    return P1S


def checks_on_amps(ifft_ssh_both, nsamp):
    zimag = np.imag(ifft_ssh_both)
    zreal = np.real(ifft_ssh_both)
    zavg = np.sum(zreal) / nsamp
    zavgsq = np.sum(zreal ** 2) / nsamp
    H13 = 4 * sqrt(zavgsq)

    max_imag = np.max(np.abs(zimag))
    return zavg, H13, max_imag



L = 100
N = 1024
z_hat, k, two_sided, SK, delta_k = compute_amplitudes(L, N)
for_plot = np.arange(0, L, (L / N))
zcomp = ifft(z_hat)
np.savetxt('real_50k_wave', np.real(zcomp))
plt.plot(for_plot, np.real(zcomp))
plt.xlabel('Distance (meters)')
plt.ylabel('SSH Elevation (meters)')
plt.title('Sea Surface Height')
plt.show()

# plt.plot(fft(zcomp),  'o', color = 'b')
# plt.plot(z_hat, 'r')
# plt.show()
p1s = compute_periodogram(np.real(zcomp), N, delta_k)
plt.loglog(SK,'b', label = 'Pierson-Moskowitz Spectra')
plt.loglog(p1s,'r', label = 'Periodogram for surface')
plt.ylim(SK.min(), p1s.max())
plt.title('Periodogram')
plt.xlabel('Spatial Frequency k (rad/m)')
plt.ylabel('Variance Density ($m^2$/(rad/m))')
plt.legend()
plt.show()

zavg, h13, max_img = checks_on_amps(zcomp, N)
