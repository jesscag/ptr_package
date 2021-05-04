"""
Generate SWH/SSH and compute correlation between observed and original
"""
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from waves2.wave_gen.two_d_spect import compute_ssh
from waves2.gauss import gauss_OBP
from waves2.ptr import ptr_run

"""
run full observation
return correlation of original to observed
save plots of spect and ssh
"""
output_dir = os.path.dirname(os.getcwd())

"""check for output directory"""
output = os.path.join(output_dir, 'output')
if os.path.isdir(output):
    print('directory exists good job ')
else:
    print('making directory')
    os.mkdir(output)
"""create output dir for day"""
current_time = datetime.datetime.now()
day, month = current_time.day, current_time.month
if not os.path.isdir(os.path.join(output, str(month) + '_' + str(day))):
    os.mkdir(os.path.join(output, str(month) + '_' + str(day)))
    print('output for this run created')
else:
    print('output day set')

"""where is additional data if needed
create data directory """
data = os.path.join(output_dir, 'data')
if not os.path.isdir(data):
    os.mkdir(data)
else:
    if os.listdir(data) == 0:
        print('no data selected')

"""
sea surface height data info 
ssh in data folder 
check dimensions 
is wave from spect created? grab it instead of making 
it 
"""
hycom = ''
wave_data = ''
if not len(hycom) == 0:
    hycom_data = np.loadtxt(os.join(data, hycom))
    xdim, ydim = hycom_data.shape
    # 1km to 5km square data needed??
if not len(wave_data) == 0:
    wave_data = np.loadtxt(os.join(data, wave_data))
    xdim, ydim = wave_data.shape

"""
gather info needed for full run
std gaussroll off, xdim, ydim, n number samp, L length 
"""
inputs = input('Final Length (meters), Nsamp, Length (meters), standard deviation (meters)')
inputs = inputs.split()
inputs = list(map(int, inputs))
t1 = datetime.datetime.now()
# generate waves from chosen spectra, L,  n
ssh_meters = compute_ssh(inputs[1], inputs[2], final_L=inputs[0])
np.savetxt('/Users/caggiano/SWOT_GIT/waves/ptr_package/data/amplitudes_1000', ssh_meters)
print(datetime.datetime.now() - t1)

# add to HYCOM or altimeter data
point_target = ptr_run.ptr_run(ssh_meters)
# del ssh_meters
np.savetxt('/Users/caggiano/SWOT_GIT/waves/ptr_package/data/point_target', point_target)
print(datetime.datetime.now() - t1)
#
# # tile waves to xdim, add to SSH, sense with PTR
# # xdim/L
ptr_tile = np.tile(point_target, (5, 5))
# del point_target

# smooth down data to 1km final size
x, y = ptr_tile.shape
smoothed = np.asarray(gauss_OBP(ptr_tile, x, y, inputs[3], inputs[0])).reshape(inputs[0], inputs[0])
#
# # correlate original SSH to final 1km data


t2 = datetime.datetime.now()
print(t2 - t1)

plt.contourf(smoothed, cmap='coolwarm')
plt.colorbar()
plt.show()
