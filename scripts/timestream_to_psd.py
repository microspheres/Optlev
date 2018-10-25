import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import os, re, time, glob
from scipy.signal import butter, lfilter, filtfilt
from scipy import signal
import bead_util as bu
from scipy.optimize import curve_fit
import glob

Fs = 1e4
NFFT = 2**19

bins = 20 # hist bins!

order = 2

p = bu.drive
p1 = bu.xi

v_cali = 1.0
v_meas = 20.0
Number_of_e = (7.76*10**14)


path_signal = r"C:\data\20180323\bead3_SiO2_15um_POL_NS\several_distances\back2_and_M2tilt\M2tilt_500\meas1"

file_list_signal = glob.glob(path_signal+"\*.h5")

def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist

file_list_signal = list_file_time_order(file_list_signal)

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y



def getdata_x_d(fname):
	_, fext = os.path.splitext( fname )
	if( fext == ".h5"):
		f = h5py.File(fname,'r')
		dset = f['beads/data/pos_data']
		dat = numpy.transpose(dset)
		Fs = dset.attrs['Fsamp']
		dat = dat * 10./(2**15 - 1)
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )
        
	x = dat[:, p1]-numpy.mean(dat[:, p1])
        
        x = butter_bandpass_filter(x, 48-10, 48+10, Fs, order)
	
	return x

def time_stream(file_list_signal):
    x = 0
    for i in file_list_signal:
        print "wait", i
        aux = getdata_x_d(i)
        aux = aux - np.mean(aux)
        x = x + aux
    return x/len(file_list_signal)

ind = 1e5
def psd(x):
    xpsd, freqs = matplotlib.mlab.psd(x-numpy.mean(x), Fs = Fs, NFFT = NFFT) 
    plt.figure()
    plt.plot(x)
    plt.ylabel("avg x axis")
    plt.figure()
    plt.loglog(freqs, xpsd)
    plt.ylabel("psd of avg x")
    plt.xlabel("freq [Hz]")
    return 0

x = time_stream(file_list_signal)
psd(x)
plt.show()
    
