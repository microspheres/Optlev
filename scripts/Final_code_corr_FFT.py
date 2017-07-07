#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import os, re, time, glob
from scipy.signal import butter, lfilter, filtfilt
from scipy import signal
import bead_util as bu
import glob

electron = 1.60E-19

d = 0.0008

Vpp = 200.0

conversion41 = -0.0832261222

conversion82 = 5.18258511759e-05

Vmeasurement_pp = 2000.0

Nucleons = 1.1E15

path_charge = r"C:\data\20170622\bead4_15um_QWP\charge11"

path_signal = r"C:\data\20170622\bead4_15um_QWP\reality_test2"

path_noise = r"C:\data\20170622\bead4_15um_QWP\charge11"

p = bu.drive

fdrive = 39.

Fs = 10000

li = 30.

ls = 200.

butterp = 1



file_list_noise = glob.glob(path_noise+"\*.h5")
file_list_signal = glob.glob(path_signal+"\*.h5")
file_list_charge = glob.glob(path_charge+"\*.h5")

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
	
	return [x, driveN, drive2W]


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


def getphase(fname, driveN, x, Fs):

    xdat = np.append(x, np.zeros( int(Fs/fdrive) ))
    corr2 = np.correlate(xdat, driveN)
    maxv = np.armax(corr2) 
	
    return maxv


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


def corr_aux(drive2WN, driveN, x, Jnoise, maxv):
    zero = np.zeros(maxv)
    shift_x = np.append(x, zero)
    shift_d = np.append(zero, driveN)
    shift_d2W = np.append(zero, drive2WN)

    fftx = np.fft.rfft(shift_x)
    fftd = np.fft.rfft(shift_d)
    fftd2W = np.fft.rfft(shift_d2W)

    jx = Jnoise

    Fi = np.argmax(fftd2W) - 5
    Fs = np.argmax(fftd2W) + 5

    boundi = 1500
    bounds = 7500


    corr = np.sum(np.conjugate(fftd[boundi:bounds])*fftx[boundi:bounds]/jx[boundi:bounds])/np.sum(np.conjugate(fftd[boundi:bounds])*fftd[boundi:bounds]/jx[boundi:bounds])
    corr = corr

    corr2W = np.sum(np.conjugate(fftd2W[Fi:Fs])*fftx[Fi:Fs]/jx[Fi:Fs])/np.sum(np.conjugate(fftd2W[Fi:Fs])*fftd2W[Fi:Fs]/jx[Fi:Fs])
    corr2W = corr2W

    return [corr, corr2W]





def get_drive(list):
    x, d, d2 = getdata_x_d(list[0])
    return [d, d2]




def do_corr(file_list_signal, maxv, Jnoise):
    corr = np.zeros(len(file_list_signal), complex)
    corr2W = np.zeros(len(file_list_signal), complex)
    a = 0
    for i in range(len(file_list_signal)):
        x, d, d2 = getdata_x_d(file_list_signal[i])
        c1, c2 = corr_aux(d2, d, x, Jnoise, maxv)
        corr[i] = (c1)
        corr2W[i] = (c2)
        a += 1.
        print "corr", a/len(file_list_signal)
    return [corr, corr2W]





def do_corr_trigger(file_list_signal, maxv, Jnoise, d, d2):
    corr = np.zeros(len(file_list_signal), complex)
    corr2W = np.zeros(len(file_list_signal), complex)
    a = 0
    for i in range(len(file_list_signal)):
        x, emp1, emp2 = getdata_x_d(file_list_signal[i])
        c1, c2 = corr_aux(d2, d, x, Jnoise, maxv)
        corr[i] = (c1)
        corr2W[i] = (c2)
        a += 1.
        print "corr", a/len(file_list_signal)
    return [corr, corr2W]

    


Jx = Jnoise(file_list_noise, 0)

corr, corr2 = do_corr(file_list_signal, 0, Jx)
### corr, corr2 = do_corr_trigger(file_list_signal, 0, Jx, drive_t, drive2_t)

print np.mean(corr)
print np.mean(corr2)



plt.show()

print "  "
print "e#/nucleon at w", ((np.real(np.mean(corr))/conversion41)*(Vpp/Vmeasurement_pp))/Nucleons
print "e#/nucleon at 2w", ((np.real(np.mean(corr))/conversion82)*(Vpp/Vmeasurement_pp))/Nucleons
