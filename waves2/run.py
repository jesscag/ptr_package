"""
cythonize in init or what?
"""

import sys
import os
import datetime

import gauss
import ptr
import wave_gen

"""
run full observation
return correlation of original to observed
save plots of spect and ssh
"""
#check for output directory
output = os.path.join(os.path.expanduser('~'),'waves/GIT/output')
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

