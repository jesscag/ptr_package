"""
One dimensional SSH creation with SWH matching
Pierson-Mowskowitz spectrum
"""

import numpy as np
from .spectra import spectra_PM
from cmath import sqrt
from scipy.fft import ifft, fft



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


def compute_ssh(L, N):
    z_hat, k, two_sided, SK, delta_k = compute_amplitudes(L, N)
    zcomp = ifft(z_hat)
    return np.real(zcomp)
