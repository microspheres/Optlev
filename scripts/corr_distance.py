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

bins = 10 # hist bins!

p = bu.drive
p1 = bu.xi

path_signal = r"C:\data\20180323\bead3_SiO2_15um_POL_NS\charge_dist\forward\dist8"

file_list_signal = glob.glob(path_signal+"\*.h5")

def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist

file_list_signal = list_file_time_order(file_list_signal)


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
	drive = dat[:, p] - numpy.mean(dat[:, p])
	
	return [x, drive]

def corr(x, drive):
    x = x -np.mean(x)
    drive = drive - np.mean(drive)
    c = np.sum(x*drive)
    return c

def plot_gauss_corr(file_list):
    C = []
    for i in file_list_signal:
        x, d = getdata_x_d(i)
        c = corr(x,d)
        C.append(c)



    s = C
    print s
    def gauss(x,a,b,c):
        g = c*np.exp(-0.5*((x-a)/b)**2)
        return g

    h,b = np.histogram(s, bins = bins)

    bc = np.diff(b)/2 + b[:-1]

    p0 = [np.mean(s), np.std(s)/np.sqrt(len(s)), 30]
    try:
        popt, pcov = curve_fit(gauss, bc, h, p0)
    except:
        popt = p0
        pcov = np.zeros([len(p0),len(p0)])


    space = np.linspace(bc[0],bc[-1], 1000)

    label_plot = str(popt[0]) + " $\pm$ " + str(np.sqrt(pcov[0,0]))

    print "result from charge fit in e#"
    print popt
    print np.sqrt(pcov[0,0])

    plt.figure()
    plt.errorbar(bc, h, yerr = np.sqrt(h), fmt = 'ko', label = label_plot)
    plt.plot(space, gauss(space,*popt))
    plt.xlabel("correlation arb units")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(path_signal,'histogram.pdf'))


plot_gauss_corr(file_list_signal)


plt.figure()
y = [173, 184, 194, 213, 233, 258, 275, 292]
yerr = [1, 0.4, 0.6, 1, 0.46, 0.43, 0.9, 0.3]
x = [1,2,3,4,5,6,7,8]
plt.errorbar(x,y, yerr = yerr, fmt = "ro")

plt.show()


