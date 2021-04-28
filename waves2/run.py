import sys
import os
import datetime
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
"""
hycom = ''

"""
gather info needed for full run
std gaussroll off, xdim, ydim, n number samp, L length 
"""
inputs = input('xdim, n, L, std')
inputs = inputs.split()
inputs = list(map(int, inputs))

# generate waves from chosen spectra, L,  n
ssh_meters = compute_ssh(inputs[1], inputs[2])

# add to HYCOM or altimeter data
point_target = ptr_run.ptr_run(ssh_meters)

# tile waves to xdim, add to SSH, sense with PTR


# smooth down data to 1km final size
# correlate original SSH to final 1km data
