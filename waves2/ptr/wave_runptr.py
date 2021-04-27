import numpy as np
import matplotlib.pyplot as plt
import ptr
import time



def create_grid(xdim, ydim):
    return np.empty(shape=(xdim, ydim))


def wave_params(Ws):
    """
    :param Ws: number of cos/sin couples
    :return: list of parameters to use to make wave grid
    """

    freq_w = np.arange(.1, 2, 0.001)  # freq (seconds)
    amp = np.arange(1, 3, 1)  # meters
    dir = np.arange(0, 2)
    params = []
    for i in range(Ws * 2):
        # k_select = wave_k[np.random.choice(len(wave_k), size=1)]
        freq_select = freq_w[np.random.choice(len(freq_w), size=1)]
        dir_select = np.random.choice(dir)
        k_select = freq_select ** 2 / 9.81
        if dir_select == 1:
            k_select = -1 * k_select
        per_select = 2 * np.pi / freq_select
        # per_select = per_T[np.random.choice(len(per_T), size=1)]
        amp_select = amp[np.random.choice(len(amp), size=1)]
        params.append([k_select, freq_select, per_select, amp_select])

    return params


def wave_form(xdim, ydim, params):
    """
    :param grid: grid for wave in meter spacing
    :param xdim: dimensions of grid
    :param ydim: dimensions of grid
    :param params: list of randomly assigned parameters for waves
    :return: grid with wave
    """
    WAVE = np.empty(shape=(xdim, ydim))
    for i in range(xdim):
        for j in range(ydim):
            w = 0
            for wave in range(0, len(params), 2):
                p = params[wave]
                p2 = params[wave + 1]
                W = p[3] * np.cos(p[0] * i - p[1] * p[2]) - p2[3] * np.sin(p2[0] * j - p2[1] * p2[2])
                # W = p[3] * np.cos(p[0] * i - p[1] * p[2]) + p[3] * np.sin(p[0] * j - p[1] * p[2])
                w += W
            WAVE[i, j] = w
    return WAVE


def wave_form_2(xdim, ydim):
    Lx = 0.01 * float(xdim)
    Ly = 0.01 * float(ydim)
    x = 0.01 * np.arange(0, xdim)
    y = 0.01 * np.arange(0, ydim)
    Z = np.empty(shape=(xdim, ydim))
    Z[:] = np.nan

    Nwavex1 = 15
    kx1 = 2.0 * np.pi * Nwavex1 / Lx
    Nwavey1 = 1
    ky1 = 2.0 * np.pi * Nwavey1 / Ly
    phi1 = np.rad2deg(np.arctan(ky1 / kx1))
    Amp1 = 1.0
    phase1 = 0

    Nwavex2 = 10;  # waves in the x direction in [0,Lx]
    kx2 = 2.0 * np.pi * Nwavex2 / Lx

    Nwavey2 = 3;  # waves in the y direction in [0,Ly]
    ky2 = 2.0 * np.pi * Nwavey2 / Ly
    phi2 = np.rad2deg(np.arctan(ky2 / kx2))
    phase2 = np.pi / 2
    Amp2 = 0.5

    for ix in range(0, xdim):
        print(ix)
        for iy in range(0, ydim):
            Z[ix, iy] = Amp1 * np.cos(kx1 * x[ix] + ky1 * y[iy] + phase1) + Amp2 * np.cos(
                kx2 * x[ix] + ky2 * y[iy] + phase2)
    return Z


if __name__ == "__main__":

    xdim = 50000
    x_for = 2
    ydim = 2
    # wave_p = 3

    start_time = time.time()
    ssh_hycom = np.loadtxt('/Users/caggiano/SWOT_GIT/waves/GIT/interp_hycom_56s')
    ssh_reshape = np.broadcast_to(ssh_hycom, (2,50000)).T

    noise = np.random.random(size=(xdim * ydim)).reshape(xdim, ydim)
    # params = wave_params(wave_p)
    fin = (wave_form_2(xdim, ydim)) * 10**2
    wave_ssh = fin + ssh_reshape

    # o = []
    # for i in range(0, xdim, x_for):
    #     # print(i)
    #     # print(fin[i:i + x_for].shape)
    #     observed = ptr.ptr_store(x_for, ydim, wave_ssh[i:i + x_for])
    #     o.append(observed)
    #
    # obs = np.asarray(o).reshape(xdim, ydim)
    # np.savetxt('obs50_56s', obs)
    # #
    # #
    # # fin_time = time.time() - start_time
