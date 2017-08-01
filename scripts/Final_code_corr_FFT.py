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

start_index = 0

electron = 1.60E-19

d = 0.0008

Vpp = 200.0

conversion41 = 0.248313748577

conversion82 = 7.24620552647e-05

Vmeasurement_pp = 200.0*7.3

Nucleons = 1.1E15

path_charge = r"C:\data\20170726\bead8_15um_QWP\steps\calibration_1positive"

path_signal = r"C:\data\20170726\bead8_15um_QWP\steps\plates_and_flips_2"

path_noise = r"C:\data\20170717\bead15_15um_QWP\steps\calibration_charge"

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
        
	x = dat[start_index:, 0]-numpy.mean(dat[start_index:, 0])
        x =  butter_bandpass_filter(x, li, ls, Fs, butterp)
	driveN = ((dat[start_index:, p] - numpy.mean(dat[start_index:, p])))/np.max((dat[start_index:, p] - numpy.mean(dat[start_index:, p])))

	driveNf = butter_bandpass_filter(driveN, li, ls, Fs, butterp)/np.max(butter_bandpass_filter(driveN, li, ls, Fs, butterp))
    
	drive2W = (driveNf*driveNf - np.mean(driveNf*driveNf))/np.max(driveNf*driveNf - np.mean(driveNf*driveNf))
	
	return [x, driveN, drive2W  ]


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
        
	x = dat[start_index:, 0]-numpy.mean(dat[start_index:, 0])

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

    boundi = np.argmax(fftd) - 5
    bounds = np.argmax(fftd) + 5


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

def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist

def find_voltagesDC(list):
    aux = np.zeros(len(list))
    for i in range(len(list)):
        aux[i] = float(re.findall("-?\d+mVDC",list[i])[0][:-4])
    return aux


def organize_DC_pos(list):
    file_list_new = []
    for i in range(len(list)):
        if float(re.findall("-?\d+mVDC",list[i])[0][:-4]) > 0:
            file_list_new.append(list[i])
    return file_list_new

def organize_DC_neg(list):
    file_list_new = []
    for i in range(len(list)):
        if float(re.findall("-?\d+mVDC",list[i])[0][:-4]) < 0:
            file_list_new.append(list[i])
    return file_list_new

def list_corr(v1,v2):
    a = []
    for i in range(len(v1)-1):
        aux = (v1[i+1]+v2[i])*0.5
        a.append(aux)
    return np.real(a)
    
file_list_pos = organize_DC_pos(list_file_time_order(file_list_signal))
file_list_neg = organize_DC_neg(list_file_time_order(file_list_signal))

l1 = len(file_list_pos)
l2 = len(file_list_neg)
lmin = np.min([l1,l2])

file_list_pos = file_list_pos[:lmin]
file_list_neg = file_list_neg[:lmin]

Jx = Jnoise(file_list_noise, 0)

drive_t, drive2_t = get_drive(file_list_charge)

# # corr, corr2 = do_corr(file_list_signal, 0, Jx)
corr, corr2 = do_corr_trigger(file_list_signal, 0, Jx, drive_t, drive2_t)


corr_pos, bpos = do_corr_trigger(file_list_pos, 0, Jx, drive_t, drive2_t)
corr_neg, bneg = do_corr_trigger(file_list_neg, 0, Jx, drive_t, drive2_t)

print np.mean(corr)
print np.mean(corr2)


print "  "
print "e#/nucleon at w", ((np.real(np.mean(corr))/conversion41)*(Vpp/Vmeasurement_pp))/(Nucleons)
print "e#/nucleon at 2w", ((np.real(np.mean(corr2))/conversion82)*(Vpp/Vmeasurement_pp))/(Nucleons)


print 0.5*((np.real(np.mean(corr_pos) + np.mean(corr_neg))/conversion41)*(Vpp/Vmeasurement_pp))/(Nucleons)

a = list_corr(corr_pos,corr_neg)

print ((np.mean(a)/conversion41)*(Vpp/Vmeasurement_pp))/(Nucleons)


plt.figure()
plt.plot(((np.real(corr_pos)/conversion41)*(Vpp/Vmeasurement_pp))/(Nucleons), 'ro')
plt.plot(((np.real(corr_neg)/conversion41)*(Vpp/Vmeasurement_pp))/(Nucleons), 'bo')
plt.plot(((np.real((corr_pos+corr_neg)/2.)/conversion41)*(Vpp/Vmeasurement_pp))/(Nucleons), 'go')
plt.plot(((a/conversion41)*(Vpp/Vmeasurement_pp))/(Nucleons), 'ko')
plt.grid()
plt.show()

plt.figure()
plt.plot(((np.real(bpos + bneg)/conversion82)*(Vpp/Vmeasurement_pp))/(Nucleons), 'ro')
plt.grid()
plt.show()
