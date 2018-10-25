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

from scipy.optimize import curve_fit

startfile = 0

start_index = 0

electron = 1.60E-19

d = 0.0008

Vpp = 200.0*0.1

conversion41 = 0.3513144

conversion82 = 0.3513144

Vmeasurement_pp = 200.0*0.1

Nucleons = 1.1E15/27.

path_charge = r"C:\data\20170905\bead4_5um_QWP_NS\calibration_1positive"

path_signal = r"C:\data\20170905\bead4_5um_QWP_NS\calibration_1positive"

path_noise = r"C:\data\20170717\bead15_15um_QWP\steps\calibration_charge"

p = bu.drive

fdrive = 39.

Fs = 10000

li = 30.

ls = 200.

butterp = 1

X = bu.xi
Y = bu.yi
Z = bu.zi


file_list_noise = glob.glob(path_noise+"\*.h5")
file_list_signal = glob.glob(path_signal+"\*.h5")[startfile:]
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




def getdata_general(fname,XX):
	_, fext = os.path.splitext( fname )
	if( fext == ".h5"):
		f = h5py.File(fname,'r')
		dset = f['beads/data/pos_data']
		dat = numpy.transpose(dset)
		Fs = dset.attrs['Fsamp']
		dat = dat * 10./(2**15 - 1)
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )
        
	x = dat[start_index:, XX]-numpy.mean(dat[start_index:, XX])
        x =  butter_bandpass_filter(x, li, ls, Fs, butterp)
	driveN = ((dat[start_index:, p] - numpy.mean(dat[start_index:, p])))/np.max((dat[start_index:, p] - numpy.mean(dat[start_index:, p])))

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
        
	x = dat[start_index:, X]-numpy.mean(dat[start_index:, X])

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

    # boundi = 1500
    # bounds = 7500

    boundi = np.argmax(fftd) - 10
    bounds = np.argmax(fftd) + 10


    corr = np.sum(np.conjugate(fftd[boundi:bounds])*fftx[boundi:bounds]/jx[boundi:bounds])/np.sum(np.conjugate(fftd[boundi:bounds])*fftd[boundi:bounds]/jx[boundi:bounds])
    corr = corr

    corr2W = np.sum(np.conjugate(fftd2W[Fi:Fs])*fftx[Fi:Fs]/jx[Fi:Fs])/np.sum(np.conjugate(fftd2W[Fi:Fs])*fftd2W[Fi:Fs]/jx[Fi:Fs])
    corr2W = corr2W

    return [corr, corr2W]





def get_drive(list):
    x, d, d2 = getdata_general(list[0],bu.xi)
    return [d, d2]




def do_corr(file_list_signal, maxv, Jnoise, XX):
    corr = np.zeros(len(file_list_signal), complex)
    corr2W = np.zeros(len(file_list_signal), complex)
    a = 0
    for i in range(len(file_list_signal)):
        x, d, d2 = getdata_general(file_list_signal[i], XX)
        c1, c2 = corr_aux(d2, d, x, Jnoise, maxv)
        corr[i] = (c1)
        corr2W[i] = (c2)
        a += 1.
        print "corr", a/len(file_list_signal)
    return [corr, corr2W]





def do_corr_trigger(file_list_signal, maxv, Jnoise, d, d2, XX):
    corr = np.zeros(len(file_list_signal), complex)
    corr2W = np.zeros(len(file_list_signal), complex)
    a = 0
    for i in range(len(file_list_signal)):
        x, emp1, emp2 = getdata_general(file_list_signal[i], XX)
        c1, c2 = corr_aux(d2, d, x, Jnoise, maxv)
        corr[i] = (c1)
        corr2W[i] = (c2)
        a += 1.
        print "corr", a/len(file_list_signal)
    return [corr, corr2W]

def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist




Jx = Jnoise(file_list_noise, 0)

drive_t, drive2_t = get_drive(file_list_charge)

corrX, corr2X = do_corr_trigger(file_list_signal, 0, Jx, drive_t, drive2_t, X)
corrY, corr2Y = do_corr_trigger(file_list_signal, 0, Jx, drive_t, drive2_t, Y)
corrZ, corr2Z = do_corr_trigger(file_list_signal, 0, Jx, drive_t, drive2_t, Z)

print np.mean(corrX)
print np.mean(corrY)
print np.mean(corrZ)
