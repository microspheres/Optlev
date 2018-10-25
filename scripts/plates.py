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
NFFT = 2**15

sleep = 5.

p = bu.drive

make_psd_plot = False

# fname0 = r"auto_xyzcool_G100_att_synth4500mV41Hz0mVdc_stage_tilt_-597thetaY_0thetaZ.h5"

path = r"C:\data\20170622\bead4_15um_QWP\dipole22_Y"

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

def get_most_recent_file(p):

    ## only consider single frequency files, not chirps
    filelist = glob.glob(os.path.join(p,"*.h5"))  ##os.listdir(p)
    #filelist = [filelist[0]]
    mtime = 0
    mrf = ""
    for fin in filelist:
        if( fin[-3:] != ".h5" ):
            continue
        f = os.path.join(path, fin) 
        if os.path.getmtime(f)>mtime:
            mrf = f
            mtime = os.path.getmtime(f)

    fnum = re.findall('\d+.h5', mrf)[0][:-3]
    return mrf#.replace(fnum, str(int(fnum)-1))

def get_position(dpsd):
    b = np.argmax(dpsd)
    return 2*b + 1

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
        peaksD[i] = dpsd[(b-1)/2]
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




def plot_peaks2F_demand(file_list):

    peaks2F = np.zeros(len(file_list))
    peaksD = np.zeros(len(file_list))
    thetaY = np.zeros(len(file_list))
    thetaZ = np.zeros(len(file_list))
    for i in range(len(file_list)):
        freqs, xpsd, dpsd = getdata1(file_list[i])
        b = get_position(dpsd)
        peaks2F[i] = xpsd[b] + xpsd[b-1] + xpsd[b+1] # all peak
        peaksD[i] = dpsd[(b-1)/2]
        f = file_list[i]
        thetaY[i] = 0
        thetaZ[i] = 0
        # m = f.rfind("stage_tilt_") + 11
        # n = f.rfind("thetaZ")
        # thetaY[i] = float(f[m:n])
        print thetaY[i], thetaZ[i]
        if make_psd_plot:
                fig = plt.figure()
                plt.loglog(freqs, xpsd)
                plt.loglog(freqs[b], xpsd[b], "x")
                plt.show()
    return [thetaY, thetaZ, np.sqrt(peaks2F), np.sqrt(peaksD), file_list[0]]

# def plot_peaks2F_placebo(file_list): # for file list with no steps
#     file_list = time_order(file_list)
#     peaks2F = np.zeros(len(file_list))
#     peaksF = np.zeros(len(file_list))
#     peaksD = np.zeros(len(file_list))
#     thetaY = np.zeros(len(file_list))
#     for i in range(len(file_list)):
#         dpsd = getdata1(file_list[i])[2]
#         xpsd = getdata1(file_list[i])[1]
#         b = get_position(dpsd)
#         peaks2F[i] = xpsd[b] + xpsd[b-1] + xpsd[b+1]
#         peaksF[i] = xpsd[(b-1)/2]
#         peaksD[i] = dpsd[(b-1)/2]
#         thetaY[i] = i
#         print thetaY[i]
#     return [thetaY, np.sqrt(peaks2F), np.sqrt(peaksD), np.sqrt(peaksF)]

# thetaY, thetaZ, peak2W, peakD = plot_peaks2F(file_list)
# # thetaY, peak2W, peakD, peakW = plot_peaks2F_placebo(file_list)

# plt.figure()
# # plt.plot(thetaY ,peak2W)
# plt.plot(thetaY ,peak2W/peakD, 'o')
# # plt.plot(thetaY ,peakW/peakD, 'o')
# plt.grid()
# plt.show()

counter = []

fig0 = plt.figure()
plt.hold(False)
last_file = []
while True:
        file_list = [get_most_recent_file( path ),]
        if( file_list[0] == last_file ): 
                continue
        else:
                last_file = file_list[0]
        time.sleep(sleep)
        Y, Z, F2, D, fl = plot_peaks2F_demand(file_list)
        print fl
        make_psd_plot = False

        counter.append(F2/D)
        plt.plot(counter)
        plt.draw()
        plt.pause(0.001)
        plt.grid()
