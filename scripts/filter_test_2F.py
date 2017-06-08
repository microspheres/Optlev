## load all files in a directory and plot the correlation of the resonse
## with the drive signal versus time

import numpy, h5py, matplotlib

import numpy as np
import matplotlib, calendar
import matplotlib.pyplot as plt
import os, re, time, glob
import bead_util as bu
import scipy.signal as sp
import scipy.optimize as opt
import cPickle as pickle
from scipy.signal import butter, lfilter, filtfilt

from scipy.optimize import curve_fit

fname0 = r"test_xyzcool_G100_att_synth5000mV41Hz0mVdc_9.h5"
path = r"C:\data\20170530\bead7_15um_QWP\test_corr"
ts = 1.

NFFT = 2**17

li = 30.

ls = 50.

# fdrive is the min freq of the freq comb
fdrive = np.min([41.])
make_plot = True

data_columns = [0, bu.xi] ## column to calculate the correlation against
drive_column = bu.drive ##-1 ## column containing drive signal



def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def test(fname):
        print "Getting phase from: ", fname 
        dat, attribs, cf = bu.getdata(os.path.join(path, fname))
        f = h5py.File(os.path.join(path, fname),'r')
        dset = f['beads/data/pos_data']
        fsamp = attribs["Fsamp"]
        xdat = dat[:,data_columns[1]]
        xdat = np.append(xdat, np.zeros( int(fsamp/fdrive) ))

        driver = butter_bandpass_filter(dat[:,drive_column], li, ls, fsamp, order=3)/np.max( butter_bandpass_filter(dat[:,drive_column], li, ls, fsamp, order=3))
        Ddriver = np.gradient(driver)/np.max(np.gradient(driver))
        driver2F = driver*Ddriver/np.max(np.abs(driver*Ddriver))
        corr2 = np.correlate(xdat,driver*Ddriver)
        maxv = np.argmax(corr2)
        
        Fs = dset.attrs['Fsamp']
        xpsd, freqs = matplotlib.mlab.psd(dat[:, drive_column]-numpy.mean(dat[:, drive_column]), Fs = Fs, NFFT = NFFT)
        driverpsd, freqs = matplotlib.mlab.psd(driver-numpy.mean(driver), Fs = Fs, NFFT = NFFT)
        Ddriverpsd, freqs = matplotlib.mlab.psd(Ddriver-numpy.mean(Ddriver), Fs = Fs, NFFT = NFFT)
        driverpsd2F, freqs = matplotlib.mlab.psd(driver2F-numpy.mean(driver2F), Fs = Fs, NFFT = NFFT)

        cf.close()

        return [driverpsd, xpsd, freqs, Ddriverpsd, driverpsd2F, driver2F, driver, dat[:,drive_column]/np.max(dat[:,drive_column])]




# plt.loglog(test(fname0)[2][300:1200],test(fname0)[3][300:1200])
# plt.loglog(test(fname0)[2][300:1200],test(fname0)[0][300:1200])
# plt.loglog(test(fname0)[2][300:1200],test(fname0)[4][300:1200])
plt.plot(test(fname0)[5][5100:5500])
plt.plot(test(fname0)[6][5100:5500])
plt.plot(test(fname0)[7][5100:5500])
plt.grid()
plt.show()
