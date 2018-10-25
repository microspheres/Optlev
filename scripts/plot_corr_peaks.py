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
from scipy.signal import butter, lfilter, filtfilt

Fs = 10e3  ## this is ignored with HDF5 files

p = bu.drive

fdrive = 47.

Fs = 10000

li = 45.

ls = 49.

butterp = 3

boundi = 1500

bounds = 7500

p = bu.drive

make_psd_plot = True

path = r"C:\data\20170823\bead2_15um_QWP_NS\dipole1"

file_list = glob.glob(path+"\*.h5")



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

def time_order(file_list):
    file_list.sort(key = os.path.getmtime)
    return file_list

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
        
	x = dat[:, 0]-numpy.mean(dat[:, 0])
        x =  butter_bandpass_filter(x, li, ls, Fs, butterp)
	driveN = ((dat[:, p] - numpy.mean(dat[:, p])))/np.max((dat[:, p] - numpy.mean(dat[:, p])))

	driveNf = butter_bandpass_filter(driveN, li, ls, Fs, butterp)/np.max(butter_bandpass_filter(driveN, li, ls, Fs, butterp))
    
	drive2W = (driveNf*driveNf - np.mean(driveNf*driveNf))/np.max(driveNf*driveNf - np.mean(driveNf*driveNf))
	
	return [x, driveNf, drive2W]


def corr_aux(drive2WN, driveN, x, Jnoise, maxv):
    zero = np.zeros(maxv)
    shift_x = np.append(x, zero)
    shift_d = np.append(zero, driveN)
    shift_d2W = np.append(zero, drive2WN)

    fftx = np.fft.rfft(shift_x)
    fftd = np.fft.rfft(shift_d)
    fftd2W = np.fft.rfft(shift_d2W)
    
    Fi = np.argmax(fftd2W) - 5
    Fs = np.argmax(fftd2W) + 5
    jx = Jnoise

    corr = np.sum(np.conjugate(fftd[boundi:bounds])*fftx[boundi:bounds]/jx[boundi:bounds])/np.sum(np.conjugate(fftd[boundi:bounds])*fftd[boundi:bounds]/jx[boundi:bounds])
    corr = corr

    corr2W = np.sum(np.conjugate(fftd2W[Fi:Fs])*fftx[Fi:Fs])
    corr2W = corr2W

    return [corr, corr2W]


def getdata_noise(fname):
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

	return [x]



def Jnoise(noise_file, maxv):
    J = []
    zero = np.zeros(maxv)
    noise_aux = []
    fftnoise = []
    a = 0
    for i in noise_file:
        noise_aux = getdata_noise(i)
        shift_N = np.append(noise_aux, zero)
        fftnoise = np.fft.rfft(shift_N)
        j = np.abs(np.conjugate(fftnoise)*fftnoise)
        J = j if J == [] else J + np.array(j)
        a += 1.
        print "PSDnoise", a/len(noise_file)
    return (J/len(noise_file))**0

Jx = Jnoise(file_list, 0)

def plot_peaks2F(file_list, Jx):
    file_list = time_order(file_list)
    corr2F = np.zeros(len(file_list))
    corrF = np.zeros(len(file_list))
    thetaY = np.zeros(len(file_list))
    thetaZ = np.zeros(len(file_list))
    for i in range(len(file_list)):
        x, d, d2 = getdata_x_d(file_list[i])
        corra, corr2a = corr_aux(d2, d, x, Jx, 0)
        corrF[i] = corra
        corr2F[i] = corr2a
        f = file_list[i]
        thetaY[i] = float(re.findall("-?\d+thetaY",f)[0][:-6])
        thetaZ[i] = float(re.findall("-?\d+thetaZ",f)[0][:-6])
        # m = f.rfind("stage_tilt_") + 11
        # n = f.rfind("thetaZ")
        # thetaY[i] = float(f[m:n])
        #thetaY[i], thetaZ[i] = 0,0
        print thetaY[i], thetaZ[i]
        corrF[i] = np.correlate(x,d)
    return [thetaY, thetaZ, corrF, corr2F]

thetaY, thetaZ, corrW, corr2W = plot_peaks2F(file_list, Jx)

plt.figure()
plt.plot(thetaZ, corrW, 'o')
plt.grid()
plt.show()
