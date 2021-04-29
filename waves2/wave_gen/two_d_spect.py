"""
Two dimensional wind generated wave creation with SWH matching
Elfouhaily et al. directional gravity-capillary spectrum
Unless stated, downwind portrayed only
"""

import numpy as np
from cmath import sqrt
from scipy.fft import ifft2, fft2, fftshift
import matplotlib.pyplot as plt
from .spectra import spectra_ECK_2
from matplotlib import ticker, colors


def compute_freq(Nx, Ny, Lx, Ly):
    kxmin = (2.0 * np.pi) / Lx
    kymin = (2.0 * np.pi) / Ly
    Nkxpos = Nx / 2
    Nkypos = Ny / 2
    Deltax = Lx / Nx
    Deltay = Ly / Ny
    Nyquistx = np.pi / Deltax
    Nyquisty = np.pi / Deltay

    Deltakx = (Nyquistx - kxmin) / (Nkxpos - 1)
    Deltaky = (Nyquisty - kymin) / (Nkypos - 1)

    kxpos = kxmin + Deltakx * np.arange(Nkxpos)
    kypos = kymin + Deltaky * np.arange(Nkxpos)

    xFFT = np.arange((Nx - 1) / 2)
    yFFT = np.arange((Ny - 1) / 2)
    kxFFT = ((2 * np.pi) / Lx) * np.hstack([xFFT, Nx / 2.0, (-Nx / 2.0) + xFFT[1:]])
    kyFFT = ((2 * np.pi) / Ly) * np.hstack([yFFT, Nx / 2.0, (-Ny / 2.0) + yFFT[1:]])

    kxMATH = np.roll(kxFFT, (Nx // 2 - 1))
    kyMATH = np.roll(kyFFT, (Ny // 2 - 1))

    return kxFFT, kyFFT, kxMATH, kyMATH, Deltaky, Deltakx


def compute_spectrum(Nx, Ny, kxFFT, kyMATH):
    psi1s = np.empty(shape=(Ny, Nx))
    psi1s[:] = np.nan
    kx1s = kxFFT[:Nx // 2 + 1]

    for ikx in range(0, Nx // 2 + 1):  # 0-33, column both need to be nonnegative
        for iky in range(Ny // 2 - 1, Ny):  # 31-62, row
            k = kx1s[ikx] ** 2 + kyMATH[iky] ** 2
            phirad = np.arctan2(kyMATH[iky], kx1s[ikx])
            psi1s[iky, ikx + (Nx // 2) - 1] = spectra_ECK_2(k, phirad, spram=True)

            if Ny / 2 <= iky <= Ny:
                psi1s[Ny - 2 - iky, ikx + Nx // 2 - 1] = psi1s[iky, ikx + Nx // 2 - 1]

    for iky2 in range(0, Ny):  # 0 - 63 rows
        for ikx2 in range(0, Nx // 2):  ##0-2 columns
            psi1s[iky2, ikx2] = psi1s[iky2, Nx - 2 - ikx2]

    psi1s[Ny // 2 - 1, Nx // 2 - 1] = 0.0

    return psi1s, np.roll(psi1s, (-Nx // 2 - 1, -Ny // 2 - 1), axis=(1, 0))


def compute_amplitudes(psi_fft, Nx, Ny, deltaky, deltakx):
    """remember to use fft order"""
    """ make sure amplitudes are self adjoint  <w, Av> = <Aw, v> """
    """ can even take compled conjj by hand?? """
    """returns in FFT order ready for fft function"""
    zhat = np.ndarray(shape=(Ny, Nx), dtype='complex_')
    zhat[:] = -99999
    c3 = 1.0 / sqrt(8)
    psi_root = c3 * np.sqrt(psi_fft * deltakx * deltaky)

    ran_real = np.empty(shape=(Ny, Nx), dtype='float')
    ran_imag = np.empty(shape=(Ny, Nx), dtype='float')
    for ikx in range(0, Nx):
        ran_real[:, ikx] = np.random.normal(size=Ny)
        ran_imag[:, ikx] = np.random.normal(size=Ny)

    # nonzero and non nyquist
    for ikx in range(1, Nx // 2):  ##halfway
        for iky in range(1, Ny):  ##all but zero
            zhat[iky, ikx] = complex(
                ran_real[iky, ikx] * psi_root[iky, ikx] + ran_real[Ny - iky, Nx - ikx] * psi_root[Ny - iky, Nx - ikx],
                ran_imag[iky, ikx] * psi_root[iky, ikx] - ran_imag[Ny - iky, Nx - ikx] * psi_root[Ny - iky, Nx - ikx])
            zhat[Ny - iky, Nx - ikx] = np.conjugate(zhat[iky, ikx])  ##symetrical for hermitian

    ikx, iky = 0, 0  # reset index
    for iky in range(1, Ny // 2):
        ikx = Nx // 2
        zhat[iky, ikx] = complex(
            ran_real[iky, ikx] * psi_root[iky, ikx] + ran_real[Ny - iky, Nx - ikx] * psi_root[Ny - iky, Nx - ikx],
            ran_imag[iky, ikx] * psi_root[iky, ikx] - ran_imag[Ny - iky, Nx - ikx] * psi_root[Ny - iky, Nx - ikx])
        zhat[Ny - iky, ikx] = np.conjugate(zhat[iky, ikx])
        ikx = 0
        zhat[iky, ikx] = complex(
            ran_real[iky, ikx] * psi_root[iky, ikx] + ran_real[Ny - iky, ikx] * psi_root[Ny - iky, ikx],
            ran_imag[iky, ikx] * psi_root[iky, ikx] - ran_imag[Ny - iky, ikx] * psi_root[Ny - iky, ikx])
        zhat[Ny - iky, ikx] = np.conjugate(zhat[iky, ikx])

    # all kx, ky = 0 and Nyquist at Ny/2
    for ikx in range(1, Nx // 2):
        iky = Ny // 2
        zhat[iky, ikx] = complex(
            ran_real[iky, ikx] * psi_root[iky, ikx] + ran_real[Ny - iky, Nx - ikx] * psi_root[Ny - iky, Nx - ikx],
            ran_imag[iky, ikx] * psi_root[iky, ikx] - ran_imag[Ny - iky, Nx - ikx] * psi_root[Ny - iky, Nx - ikx])
        zhat[iky, Nx - ikx] = np.conjugate(zhat[iky, ikx])

        iky = 0
        zhat[iky, ikx] = complex(
            ran_real[iky, ikx] * psi_root[iky, ikx] + ran_real[iky, Nx - ikx] * psi_root[iky, Nx - ikx],
            ran_imag[iky, ikx] * psi_root[iky, ikx] - ran_imag[iky, Nx - ikx] * psi_root[iky, Nx - ikx])
        zhat[iky, Nx - ikx] = np.conjugate(zhat[iky, ikx])

    # nyquist's
    iky, ikx = Ny // 2, 0
    zhat[iky, ikx] = complex(
        ran_real[iky, ikx] * psi_root[iky, ikx] + ran_real[Ny - iky, ikx] * psi_root[Ny - iky, ikx],
        ran_imag[iky, ikx] * psi_root[iky, ikx] - ran_imag[Ny - iky, ikx] * psi_root[Ny - iky, ikx])

    iky, ikx = Ny // 2, Nx // 2
    zhat[iky, ikx] = complex(
        ran_real[iky, ikx] * psi_root[iky, ikx] + ran_real[Ny - iky, Nx - ikx] * psi_root[Ny - iky, Nx - ikx],
        ran_imag[iky, ikx] * psi_root[iky, ikx] - ran_imag[Ny - iky, Nx - ikx] * psi_root[Ny - iky, Nx - ikx])
    iky, ikx = 0, Ny // 2
    zhat[iky, ikx] = complex(
        ran_real[iky, ikx] * psi_root[iky, ikx] + ran_real[iky, Nx - ikx] * psi_root[iky, Nx - ikx],
        ran_imag[iky, ikx] * psi_root[iky, ikx] - ran_imag[iky, Nx - ikx] * psi_root[iky, Nx - ikx])
    zhat[0, 0] = complex(0, 0)

    return zhat * Nx * Ny


def plot_spect(psi, xmath, ymath):
    Psi1splot = psi[:, n // 2 - 1:]
    Psi1splot2 = Psi1splot
    xmath_plot = xmath[n // 2 - 1:]
    ymath_plot = ymath
    min_e, max_e = np.exp(-20), 10

    Psi1splot = np.ma.masked_where(Psi1splot < min_e, Psi1splot)

    fig, ax = plt.subplots()
    # cs = ax.contourf(xmath_plot, ymath_plot, Psi1splot, locator = ticker.LogLocator() , levels = 10)
    lev_exp = np.arange(np.floor(np.log10(np.nanmin(Psi1splot)) - 1), np.ceil(np.log10(np.nanmax(Psi1splot)) + 2))
    levs = np.power(10, lev_exp)
    cs = ax.contourf(xmath_plot, ymath_plot, Psi1splot2,
                     levs, norm=colors.LogNorm(), extend='both', cmap='coolwarm')
    plt.title('m^2/(rad/m)^2')
    fig.colorbar(cs)
    plt.show()


def compute_ssh(n, L, final_L):
    xfft, yfft, xmath, ymath, delkx, delky = compute_freq(n, n, L, L)
    print('freq made')
    psi, fft_psi = compute_spectrum(n, n, xfft, ymath)
    print('psi filled')
    ssh = []
    for i in range(final_L):
        zhat = compute_amplitudes(psi, n, n, delky, delkx)  ##zhat giving nan??
        print('amplitudes created')
        a = np.real(zhat)
        comp = ifft2(zhat)
        ssh.append(np.real(comp))
    return ssh

#
# if __name__ == "__main__":
#     # t1 = time.time()
#     # n = 64*50
#     # L = 100*50
#     # xfft,yfft,xmath,ymath, delkx, delky = compute_freq(n,n,L,L)
#     # psi, fft_psi = compute_spectrum(n,n, xfft, ymath)
#     # zhat = compute_amplitudes(psi, n, n, delky, delkx)  ##zhat giving nan??
#     # a = np.real(zhat)
#     # comp = ifft2(zhat)
#     # print(time.time() - t1)
#     #
#     # """plot"""
#     # lev = np.linspace(np.min(np.real(comp)), np.max(np.real(comp)))
#     # plt.contourf(np.arange(L, step = L/n),np.arange(L, step = L/n), np.real(comp), levels =lev, cmap = 'coolwarm')
#     # plt.title('ssh (meters)')
#     # plt.xlabel('meters')
#     # plt.ylabel('meters')
#     # plt.colorbar()
#     # plt.show()
#     # plot_spect(psi, xmath, ymath)
#     # print(time.time() - t1)
