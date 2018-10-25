import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import bead_util as bu
import glob
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import cPickle as pickle

from correlation_each_freq_of_comb_main import *
from rotation_peak_finder_new_main import *

############### for charge measurement:

freq_list = [48.]

path_charge = r"C:\data\20180323\bead3_SiO2_15um_POL_NS\calibration\1e"

path_signal = r"C:\data\20180323\bead3_SiO2_15um_POL_NS\meas5_spin_scan"

path_noise = r"C:\data\20180323\bead3_SiO2_15um_POL_NS\meas5_spin_scan"

endfile = -1

startfile = 0

start_index = 0
end_index = -1

file_list_signal = glob.glob(path_signal+"\*.h5")
file_list_charge = glob.glob(path_charge+"\*.h5")
file_list_noise = glob.glob(path_noise+"\*.h5")

def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist

file_list_signal = list_file_time_order(file_list_signal)

file_list_signal = file_list_signal[startfile:endfile]

path_list = [path_signal]

remake_file = True
file_list1 = []
for path1 in path_list:

    file_list = glob.glob(path1+"\*.h5")
    file_list = list_file_time_order(file_list)
    file_list1 += file_list[startfile:endfile]


#things below are not to change!
d = drive(file_list_charge,1)
arg = arg_each_freq(d, freq_list)
xt = xtemplate_charge(file_list_charge,arg)
jpsd_arg = jnoise(file_list_noise, arg)
cali = auto_calibration(xt, jpsd_arg)
#things up are not to change!


corr = corr(xt, file_list_signal, jpsd_arg, arg)
corrfreq = corr_allfreq(corr)
corrfreq = np.array(corrfreq)


correlation = np.real(corrfreq*cali)



####################
#######rotation
NFFT = 2**17
parameters = [517000., 0.1, 5000, "none", NFFT]  # last entry shoulp help with aliasing. Gets "up", "down" or "none". "up" for curves that go up," down" for curves that go down and "none" for nothing to happens.


c1 = finder(file_list1, parameters)

rotation = np.array(c1[0])



### plot

plt.figure()
plt.plot(rotation/1.0e6 ,correlation, "ro")
plt.xlabel("Rotation freq [MHz]")
plt.ylabel("Charge in e#")
plt.savefig(os.path.join(path_signal,'background_vs_spin.pdf'))
plt.show()
