#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 13:28:01 2017

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
import time

Fs = 10e3  ## this is ignored with HDF5 files
NFFT = 2**19

sleep = 5.

p = bu.drive

make_psd_plot = True

# fname0 = r"auto_xyzcool_G100_att_synth4500mV41Hz0mVdc_stage_tilt_-597thetaY_0thetaZ.h5"

path = r"C:\data\20170717\bead15_15um_QWP\dipole18_Y"
file_list = glob.glob(path+"\*.h5")

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
        
	x = dat[:, 0]-numpy.mean(dat[:, 0])
	xpsd, freqs = matplotlib.mlab.psd(x, Fs = Fs, NFFT = NFFT)
	drive = ((dat[:, p] - numpy.mean(dat[:, p])))/np.max((dat[:, p] - numpy.mean(dat[:, p])))
	drivepsd, freqs = matplotlib.mlab.psd(drive, Fs = Fs, NFFT = NFFT)

	return [freqs, xpsd, drivepsd]

def time_order(file_list):
    file_list.sort(key = os.path.getmtime)
    return file_list

def get_position(dpsd):
    b = np.argmax(dpsd)
    return b

def plot_peaks2F(file_list):
    file_list = time_order(file_list)
    peaks2F = np.zeros(len(file_list))
    peaksD = np.zeros(len(file_list))
    thetaY = np.zeros(len(file_list))
    thetaZ = np.zeros(len(file_list))
    if make_psd_plot: plt.figure()
    for i in range(len(file_list)):
        freqs, xpsd, dpsd = getdata1(file_list[i])
        b = get_position(dpsd)
        peaks2F[i] = xpsd[b] + xpsd[b-1] + xpsd[b+1] # all peak
        peaksD[i] = dpsd[b]
        f = file_list[i]
        thetaY[i] = float(re.findall("-?\d+thetaY",f)[0][:-6])
        thetaZ[i] = float(re.findall("-?\d+thetaZ",f)[0][:-6])
        # m = f.rfind("stage_tilt_") + 11
        # n = f.rfind("thetaZ")
        # thetaY[i] = float(f[m:n])
        print thetaY[i], thetaZ[i]
        if make_psd_plot:
                plt.loglog(freqs, xpsd)
                plt.plot(freqs[b], xpsd[b], "x")
    return [thetaY, thetaZ, np.sqrt(peaks2F), np.sqrt(peaksD)]

thetaY, thetaZ, peak2W, peakD = plot_peaks2F(file_list)
# thetaY, peak2W, peakD, peakW = plot_peaks2F_placebo(file_list)

plt.figure()
# plt.plot(thetaY ,peak2W)
plt.plot(thetaY ,peak2W/peakD, 'o')
# plt.plot(thetaY ,peakW/peakD, 'o')
plt.grid()
plt.show()
