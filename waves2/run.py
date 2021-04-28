import sys
import os
import datetime
from waves2 import gauss, ptr, wave_gen


"""
run full observation
return correlation of original to observed
save plots of spect and ssh
"""
output_dir = os.path.dirname(os.getcwd())

#check for output directory
output = os.path.join(output_dir,'output')
if os.path.isdir('output'):
    print('directory exists good job ')
else:
    print('making directory')
    os.mkdir('output')
#create output dir for day
current_time = datetime.datetime.now()
day, month = current_time.day, current_time.month
if not os.path.isdir('output/'+str(month)+'_'+str(day)):
    os.mkdir('output/'+str(month)+'_'+str(day))
    print('output for this run created')
else:
    print('output day set')

"""
gather info needed for full run
std gaussroll off, xdim, ydim, n number samp, L length 
"""

inputs = input('xdim, L, n, std,spect')
inputs = inputs.split()


##generate waves from chosen spectra, L,  n
## tile waves to xdim, add to SSH, sense with PTR
## smooth down data to 1km final size
## correlate original SSH to final 1km data