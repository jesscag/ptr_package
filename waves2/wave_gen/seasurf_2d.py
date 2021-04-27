import numpy as np
from spectra import spectra_PM
from cmath import sqrt
from scipy.fft import ifft, fft, fftshift
import matplotlib.pyplot as plt

Nx, Ny = 8, 8
Deltax = 0.1
Deltay = 0.1
Lx = Deltax*float(Nx)
Ly = Deltax*float(Ny)

x = Deltax * np.arange(0, Nx)
y = Deltay * np.arange(0, Ny)
Z = np.empty(shape = (Nx, Ny))
Z[:] = np.nan

##frequncies
kxmin = 2 * np.pi/Lx
kymin = 2 * np.pi/Ly
minxwave = 2.0*Deltax
minywave = 2.0*Deltay
Nyquistx = np.pi/Deltax
Nyquisty = np.pi/Deltay
kxmax = Nyquistx
kymax = Nyquisty

#compute positive frequencies
Nkxpos = Nx/2
Nkypos = Ny/2
dkx = (kxmax - kxmin)/float(Nkxpos-1)
dky = (kymax - kymin)/float(Nkypos-1)
kxpos = kxmin + dkx*np.arange(0, Nkxpos)
kypos = kymin + dky*np.arange(0, Nkypos)

#reorder freq for FFT order
xFFT = np.arange(0,Nx/2-1) + 1.0
yFFT = np.arange(0,Ny/2-1) + 1.0
kxFFT = (2.0*np.pi/Lx)*np.hstack([0.0, xFFT, Nx/2.0, -Nx/2.0+xFFT])
kyFFT = (2.0*np.pi/Ly)*np.hstack([0.0, yFFT, Ny/2.0, -Ny/2.0+yFFT])
##math order
# kxmath = fftshift(kxFFT)
# kymath = fftshift(kyFFT)  makes nyquist 0th element
kxmath = np.hstack([-kxpos[::-1][1:], 0 , kxpos])
kymath = np.hstack([-kypos[::-1][1:], 0 , kypos])


#make spatial Z
Nwavex1 = 2
kx1 = 2.0 *np.pi * Nwavex1 / Lx
Nwavey1 = 1
ky1 = 2.0 *np.pi * Nwavey1 / Ly
phi1 = np.rad2deg(np.arctan(ky1/kx1))
Amp1 = 1.0
phase1 = 0

Nwavex2 = 4;  # waves in the x direction in [0,Lx]
kx2 = 2.0 *np.pi * Nwavex2 / Lx

Nwavey2 = 3;  # waves in the y direction in [0,Ly]
ky2 = 2.0 *np.pi * Nwavey2 / Ly
phi2 = np.rad2deg(np.arctan(ky2/kx2))
phase2 = np.pi / 2
Amp2 = 0.5

for ix in range(0, Nx):
    for iy in range(0, Ny):
        Z[ix,iy] = Amp1*np.cos(kx1*x[ix] + ky1*y[iy] + phase1) + Amp2*np.cos(kx2*x[ix] + ky2*y[iy] + phase2)
Z = Z - np.nanmean(Z)
plt.contourf(Z, levels = 100)
plt.colorbar()
plt.show()


#make Z complex
Z = np.asarray(Z, dtype = 'complex128')
zhat = fft(Z,Nx)/(Nx*Ny)   ##FFT order, for plot shift back to math order
Zreal = np.real(zhat)
Zimag = np.imag(zhat)
kx,ky = np.meshgrid(kxmath, kymath)
# plt.contourf(kx,ky,fftshift(Zimag), levels = 100)
# plt.plot(kx,ky, 'o', color = 'gray')
# plt.colorbar()
# plt.show()
#
# plt.contourf(kx,ky,fftshift(Zreal), levels = 100 )
# plt.plot(kx,ky, 'o', color = 'gray')
# plt.colorbar()
# plt.show()

##go back
zBack = np.empty_like(zhat)
for i in range(0, Nx):
    for j in range(0, Ny):
        zBack[i,j] = complex(Zreal[i,j])
z_knot = ifft(zBack,Nx) * (Nx * Ny)
plt.contourf(np.real(z_knot), levels = 100)
plt.colorbar()
plt.show()
