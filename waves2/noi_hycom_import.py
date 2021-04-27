import glob as glob
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt
from netCDF4 import Dataset


def spectrum1(h, dt=1):
    """
    First cut at spectral estimation: very crude.

    Returns frequencies, power spectrum, and
    power spectral density.
    Only positive frequencies between (and not including)
    zero and the Nyquist are output.
    """
    nt = len(h)
    npositive = nt // 2
    pslice = slice(1, npositive)
    freqs = np.fft.fftfreq(nt, d=dt)[pslice]
    ft = np.fft.fft(h)[pslice]
    psraw = np.abs(ft) ** 2
    # Double to account for the energy in the negative frequencies.
    psraw *= 2
    # Normalization for Power Spectrum
    psraw /= nt ** 2
    # Convert PS to Power Spectral Density
    psdraw = psraw * dt * nt  # nt * dt is record length
    return freqs, psraw, psdraw


def bilinear_interp(along_lat, along_lon, LAT, LON, SLA):
    """
    Bilinear interpolation in space not time
    :param along_lat: along track lat
    :param along_lon: along track lon
    :param LAT: gridded lat
    :param LON: gridded lon
    :param SLA: gridded SLA
    :return: interpolated SLA to groundtrack
    """
    # grid_diff = 0.08
    YY = np.zeros_like(along_lat)
    YY[:] = np.nan
    for i in range(0, along_lat.size):
        coords_along = [along_lat[i], along_lon[i]]
        lat_loc = np.abs(LAT - coords_along[0]).argmin()
        lon_loc = np.abs(LON - coords_along[1]).argmin()
        diff = LAT[lat_loc] - coords_along[0]
        diff_lon = LON[lon_loc] - coords_along[1]
        if diff < 0:
            lat_loc1 = lat_loc
            lat_loc2 = lat_loc + 1
            # if not (LAT[lat_loc1] < coords_along[0] < LAT[lat_loc2]):
            #     print('not satisfied1 ' + str(i))
        if diff >= 0:
            lat_loc1 = lat_loc - 1
            lat_loc2 = lat_loc
            # if not (LAT[lat_loc1] < coords_along[0] < LAT[lat_loc2]):
            #     print('not satisfied2 ' + str(i))

        if diff_lon < 0:
            lon_loc1 = lon_loc
            lon_loc2 = lon_loc + 1
            # if not (LON[lon_loc1] < coords_along[1] < LON[lon_loc2]):
            #     print('not sat_lon')
        if diff_lon >= 0:
            lon_loc1 = lon_loc - 1
            lon_loc2 = lon_loc

        value_sla = np.asarray([SLA[lat_loc1, lon_loc1], SLA[lat_loc2, lon_loc1],
                                SLA[lat_loc2, lon_loc2], SLA[lat_loc1, lon_loc2]])

        u = (coords_along[1] - LON[lon_loc1]) / (LON[lon_loc2] - LON[lon_loc1])

        t = (coords_along[0] - LAT[lat_loc1]) / (LAT[lat_loc2] - LAT[lat_loc1])

        y = (1 - t) * (1 - u) * value_sla[0] + t * (1 - u) * value_sla[1] + t * u * value_sla[2] \
            + (1 - t) * u * value_sla[3]
        YY[i] = y

    return YY


def average_hycom():
    path = '/Volumes/DATA/hycom_08/1996*'
    list_files = natsorted(glob.glob(path))
    list_means = []
    for i in list_files: #year
        surf_annual = Dataset(i).variables['surf_el']
        for j in range(0, surf_annual.shape[0]):  ##time index
            print(j , np.nanmean(surf_annual[j]))
            list_means.append(np.nanmean(surf_annual[j]))
    return np.asarray(list_means)


'''
NOI data in CM 
HYCOM in Meters 
'''

# data = np.loadtxt('/Volumes/DATA/DT_noi/asc_c104_dt_noi_j2.txt', skiprows=1)
# data[data == -999.0] = np.nan
# noi_ssh = data[:, 3:]
# noi_lat = data[:, 1]
# noi_lon = data[:, 0]

noi_lat = np.empty(shape = (50000))
noi_lat[:] = -56
x = 200000
noi_lon = np.arange(0,360, 0.00002)  ##1 ish meter apart
noi_lon = noi_lon[x:x+50000]

hycom_data = Dataset('/Volumes/DATA/2012.nc4')  ##
hycom_lat = np.asarray(hycom_data.variables['lat'][:])
hycom_sl = hycom_data.variables['surf_el'][10]  # time, lat,lon turn to CM
hycom_data.close()
hycom_lon = np.arange(0, 360, 0.08)

hycom_sl_cm = hycom_sl * 10 ** 2  ##in cm
# hyc_mean = average_hycom() * 10 ** 2  ##in cm

yy = bilinear_interp(noi_lat, noi_lon, hycom_lat, hycom_lon, hycom_sl_cm)
np.savetxt('interp_hycom_56s', yy)


# noi_ssh = np.nanmean(noi_ssh, axis = 1 )
# no_nan_along = noi_ssh[~(np.isnan(noi_ssh))]
# no_nan_interp = yy[~(np.isnan(yy))]
# #
# f, ps, psd = spectrum1(no_nan_interp, dt = 6.5)
# f2, ps2, psd2 = spectrum1(no_nan_along, dt = 6.5)
# plt.loglog(f, psd, label='hycom interp', color = 'red')
# plt.loglog(f2, psd2, label='NOI one pass', color = 'blue')
# plt.legend()
# plt.xlabel('freq')
# plt.ylabel('power spect density')
# plt.savefig('spect_onepass')
# plt.show()

plt.plot(yy)
plt.show()
