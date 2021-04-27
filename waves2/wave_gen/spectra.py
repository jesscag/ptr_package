import numpy as np
from cmath import sqrt
import matplotlib.pyplot as plt
from scipy.special import gamma


def spectra_PM(k, u10):
    """
    :param k: angular freq
    :param u10: wind speed m/s
    :return: PM spect
    """
    alpha = 0.0081
    beta = 0.74
    g = 9.82
    u19 = 1.026 * u10

    C1 = 2 * k ** 3
    C2 = (g / k) ** 2
    SPM = alpha / C1 * np.exp(-1 * beta * C2 * (1 / u19 ** 4))
    return SPM


def spectra_ECKV(kpos, phirad, u10, age=1, spram=False):
    """
    :param age: developed sea (0.84 - 5)
        0.84 fully developed
        1 mature
        2-5 young
    :param kpos: angular wave number (rad/m)
    :param phirad: angle in rad (-pi to pi, 0 to 2pi)
    :param u10: wind velocity 10 meters up
    :param spram: spreading parameter
    :return: one sided energy spect
    """

    """sea state params"""
    g = 9.82
    omegac = age  # for fully developed sea, 1 for mature sea
    ustar = np.sqrt(0.00144) * u10  # u10 to friction velocity
    ao = 0.1733
    ap = 4
    cm = 0.23
    am = 0.13 * ustar / cm
    km = 370  # gravity/capilary wave boundary
    gamma = 1.7

    sigma = 0.08 * (1.0 + 4.0 * omegac ** (-3))
    twosig2 = 1.0 / (2.0 * sigma * sigma)
    alphap = 0.006 * omegac ** 0.55

    if ustar <= cm:
        alpham = 0.01 * (1.0 + np.log(ustar / cm))
    else:
        alpham = 0.01 * (1.0 + 3 * np.log(ustar / cm))

    ko = g / (u10 ** 2)
    kp = ko * omegac ** 2
    wavemax = 2.0 * np.pi / kp
    cp = np.sqrt(g / kp)
    capomega = omegac

    """Evaluate spectrum"""
    nk = np.size(kpos)
    nphi = np.size(phirad)
    E2D = np.empty(shape=(nk, nphi))
    for ku in range(0, nk):
        for phiv in range(0, nphi):
            # k = kpos[ku]
            # phi = phirad[phiv]
            k = kpos
            phi = phirad
            c = np.sqrt((g / k) * (1 + (k / km) ** 2))
            sqrtkkp = np.sqrt(k / kp)
            capgamma = np.exp(-twosig2 * (sqrtkkp - 1.0) ** 2)
            fJp = gamma ** capgamma
            fLpm = np.exp(-1.25 * (kp / k) ** 2)
            Fp = fLpm * fJp * np.exp(-0.3162 * capomega * (sqrtkkp - 1.0))
            Bl = 0.5 * alphap * (cp / c) * Fp

            Fm = fLpm * fJp * np.exp(-0.25 * (k / km - 1.0) ** 2)
            Bh = 0.5 * alpham * (cm / c) * Fm

            deltak = np.tanh(ao + ap * (c / cp) ** 2.5 + am * (cm / c) ** 2.5)

            if (phi >= -0.5 * np.pi) & (phi <= 0.5 * np.pi):

                E2D[ku, phiv] = (1.0 / np.pi) * (Bl + Bh) / k ** 4 * (1.0 + deltak * np.cos(2.0 * phi))
            else:
                E2D[ku, phiv] = 0
    return E2D


def spectra_ECK_2(k, phirad, u10=5, age=1, spram=False):
    g = 9.82
    Omegac = age
    ustar = np.sqrt(0.00144) * u10  # convert to friction velocity
    ao = 0.1733
    ap = 4.0
    cm = 0.23
    am = 0.13 * ustar / cm
    km = 370

    if Omegac <= 1:
        gammac = 1.7
    else:
        gammac = 1.7 + 6.0 * np.log10(Omegac)
    sigma = 0.08 * (1 + 4 * Omegac ** (-3))
    twosig2 = 1 / (2 * sigma ** 2)
    alphap = 0.006 * Omegac ** (0.55)

    if ustar <= cm:
        alpham = 0.01 * (1 + np.log(ustar / cm))
    else:
        alpham = 0.01 * (1 + 3 * np.log(ustar / cm))

    ko = g / (u10 ** 2)
    kp = ko * Omegac ** 2

    wavemax = 2 * np.pi / kp
    cp = np.sqrt(g / kp)
    Capomega = Omegac

    c = np.sqrt((g / k) * (1.0 + (k / km) ** 2))

    sqrtkp = np.sqrt(k / kp)

    ##low freq/long wave part

    Capgamma = np.exp(-twosig2 * (sqrtkp - 1) ** 2)
    fJp = gammac ** Capgamma
    if k == 0:
        fLpm = np.exp(0)
    else:
        fLpm = np.exp(-1.25 * (kp / k) ** 2)
    Fp = fLpm * fJp * np.exp(-0.3162 * Capomega * (sqrtkp - 1))
    Bl = 0.5 * alphap * (cp / c) * Fp

    # High freq/short wave part
    Fm = fLpm * fJp * np.exp(-0.25 * (k / km - 1) ** 2)
    Bh = 0.5 * alpham * (cm / c) * Fm

    if spram == False:
        S = 1
        Deltak = np.tanh(ao + ap * (c / cp) ** 2.5 + am * (cm / c) ** 2.5)
        Spread = 0.5 * (1.0 + Deltak * np.cos(2.0 * phirad)) / (2.0 * np.pi)

    if spram == True:
        S = 10 ##adjust this
        Cnorm = (0.5 / sqrt(np.pi))*gamma(S + 1.0) / gamma(S + 0.5)
        Spread = Cnorm * (np.cos(0.5 * phirad)) ** (2.0 * S)
        if np.abs(np.abs(phirad - np.pi)) < 1**-6:
            Spread = 0

    E2D = ((Bl + Bh) / k**4) * Spread

    return E2D





















