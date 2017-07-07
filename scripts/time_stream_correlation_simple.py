#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 11:31:01 2017

@author: fernandomonteiro
"""

import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import os, re, time, glob
from scipy.signal import butter, lfilter, filtfilt
from scipy import signal
import glob
import bead_util as bu


Fs = 10e3  ## this is ignored with HDF5 files
NFFT = 2**17

# fname0 = r"auto_xyzcool_G200_synth1000mV41Hz0mVdc_0.h5"

path = r"C:\data\20170622\bead4_15um_QWP\dipole5_Y"

tukey = 0.03

butterp = 3

li = 27.

ls = 53.

p = bu.drive

fdrive = 41.

def step(X, y, z):
    result = np.zeros(len(X)) 
    for i in np.arange(len(X)):
        if X[i] < y or X[i] > z - y:
            result[i] = 0.0        
        else:
            result[i] = 1.0
    return result


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def getdata1(fname):
	_, fext = os.path.splitext( fname )
	if( fext == ".h5"):
		f = h5py.File(fname,'r')
		dset = f['beads/data/pos_data']
		dat = numpy.transpose(dset)
		Fs = dset.attrs['Fsamp']
		dat = dat * 10./(2**15 - 1)
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )
        
	xpsd, freqs = matplotlib.mlab.psd(dat[:, 0]-numpy.mean(dat[:, 0]), Fs = Fs, NFFT = NFFT)
	x = dat[:, 0]-numpy.mean(dat[:, 0])

	dataux = dat[:, p]

	x_filt = (butter_bandpass_filter(x, li, ls, Fs, butterp))/(np.max(butter_bandpass_filter(x, li, ls, Fs, butterp)))
	x_box = (signal.tukey(len(dataux),tukey)*x_filt)/(np.max(signal.tukey(len(dataux),tukey)*x_filt))
    
	drive = ((dat[:, p] - numpy.mean(dat[:, p])))/np.max((dat[:, p] - numpy.mean(dat[:, p])))
	drivepsd, freqs = matplotlib.mlab.psd(drive, Fs = Fs, NFFT = NFFT)
    
	drivefilt = (butter_bandpass_filter(drive, li, ls, Fs, butterp))/(np.max(butter_bandpass_filter(drive, li, ls, Fs, butterp)))
    
	drivefiltbox = (signal.tukey(len(dataux),tukey)*drivefilt)/(np.max(signal.tukey(len(dataux),tukey)*drivefilt))
	drivefiltboxpsd, freqs = matplotlib.mlab.psd(drivefiltbox, Fs = Fs, NFFT = NFFT)
    
	aux = drivefilt*drivefilt - np.mean(drivefilt*drivefilt)
	auxfilt = (butter_bandpass_filter(aux, 2.*li, 2.*ls, Fs, butterp))/(np.max(butter_bandpass_filter(aux, 2.*li, 2.*ls, Fs, butterp)))
    
	drive2box = (auxfilt*signal.tukey(len(dataux),tukey))/(np.max(auxfilt*signal.tukey(len(dataux),tukey)))
	drive2boxpsd, freqs = matplotlib.mlab.psd(drive2box, Fs = Fs, NFFT = NFFT)
	
	return [freqs, drive, drivepsd, drivefiltbox, drivefiltboxpsd, drive2box, drive2boxpsd, x, xpsd, dataux, x_box]

def getphase(fname, driveN, x, Fs):

	xdat = np.append(x, np.zeros( int(Fs/fdrive) ))
	corr2 = np.correlate(xdat, driveN)
	maxv = np.argmax(corr2) 
	
#	cf.close()
	
	print (maxv)
	return maxv

def correlation(driveN, x, maxv):
	zero = np.zeros(maxv)
	shift_x = np.append(x, zero)
	shift_D = np.append(zero, driveN)
    
	corr = np.sum(shift_x*shift_D)/np.sum(shift_D*shift_D)

	return corr


file_list = glob.glob(path+"\*.h5")

def time_order(file_list):
    file_list.sort(key = os.path.getmtime)
    return file_list
    
def full_correlation_TS(file_list):
    file_list = time_order(file_list)
    thetaY = np.zeros(len(file_list))
    corr = np.zeros(len(file_list))
    for i in range(len(file_list)):
        x = getdata1(file_list[i])[7]
        x_box = getdata1(file_list[i])[10]
        driveN = getdata1(file_list[i])[1]
        driveN_box = getdata1(file_list[i])[3]
        drive2W_box = getdata1(file_list[i])[5]
        maxv = getphase(file_list[i], driveN, x, Fs)
        corr[i] = correlation(drive2W_box, x_box, maxv)
        f = file_list[i]
        m = f.rfind("stage_tilt_") + 11
        n = f.rfind("thetaY")
        thetaY[i] = float(f[m:n])
    return [thetaY, corr]


plt.figure()
plt.plot(full_correlation_TS(file_list)[0], full_correlation_TS(file_list)[1])
plt.show()


#x = getdata1(fname0)[7]
#x_box = getdata1(fname0)[10]
#driveN = getdata1(fname0)[1]
#driveN_box = getdata1(fname0)[3]
#drive2W_box = getdata1(fname0)[5]
#
#maxv = getphase(fname0, driveN, x, Fs)
#
#
#corr = correlation(drive2W_box, x_box, maxv)


#plt.plot(drive2W_box)
#plt.plot(driveN_box)
