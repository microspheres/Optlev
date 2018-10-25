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

def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist

startfile = 0

endfile = -1

start_index = 0

electron = 1.60E-19

d = 0.001

Vpp = 200.0*0.2

conversion41 = 35.60485

conversion82 = 35.60485

Vmeasurement_pp = 200.0*15.

Nucleons = (3.55e15)/(3.6)

# path_charge = r"C:\data\20170622\bead4_15um_QWP\reality_test3" #41drive
path_charge = r"C:\data\20171004\bead9_15um_QWP_NS\calibration1e\7"

path_signal = r"C:\data\20171004\bead9_15um_QWP_NS\meas\DC_no_AC"

path_noise = r"C:\data\20171004\bead9_15um_QWP_NS\meas\DC_no_AC"

p = bu.drive

fdrive = 30.

Fs = 10000

li = 10.

ls = 3300.

# li = 950.

# ls = 1050.


butterp = 1



file_list_noise = glob.glob(path_noise+"\*.h5")
file_list_signal = glob.glob(path_signal+"\*.h5")
file_list_charge = glob.glob(path_charge+"\*.h5")

# def get_specific_DC(list):
#     file_list_new = []
#     for i in range(len(list)):
#         if np.abs(float(re.findall("-?\d+mVDC",list[i])[0][:-4])) == 1880:
#             file_list_new.append(list[i])
#     return file_list_new

# file_list_signal = get_specific_DC(file_list_signal)

file_list_signal = list_file_time_order(file_list_signal)

file_list_signal = file_list_signal[startfile:endfile]

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
        
	x = dat[start_index:, 0]-numpy.mean(dat[start_index:, 0])

	return [x]


# def getphase(fname, driveN, x, Fs):

#     xdat = np.append(x, np.zeros( int(Fs/fdrive) ))
#     corr2 = np.correlate(xdat, driveN)
#     maxv = np.armax(corr2) 
	
#     return maxv

def getphase(driveN, x, Fs):

    xdat = np.append(x, np.zeros( int(Fs/fdrive) ))
    corr2 = np.correlate(xdat, driveN)
    maxv = np.argmax(corr2) 
	
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

    Fi = 0
    Fs = -1

    # Fi = np.argmax(fftd2W) - 5
    # Fs = np.argmax(fftd2W) + 5

    boundi = 0
    bounds = -1

    # boundi = np.argmax(fftd) - 5
    # bounds = np.argmax(fftd) + 5


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



def find_voltagesDC(list):
    aux = np.zeros(len(list))
    for i in range(len(list)):
        aux[i] = float(re.findall("-?\d+mVDC",list[i])[0][:-4])
    return aux


# def organize_DC_pos(list):
#     file_list_new = []
#     for i in range(len(list)):
#         if True:# float(re.findall("-?\d+mVDC",list[i])[0][:-4]) > 0:
#             file_list_new.append(list[i])
#     return file_list_new

# def organize_DC_neg(list):
#     file_list_new = []
#     for i in range(len(list)):
#         if True: # float(re.findall("-?\d+mVDC",list[i])[0][:-4]) < 0:
#             file_list_new.append(list[i])
#     return file_list_new

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
    
# file_list_pos = organize_DC_pos(list_file_time_order(file_list_signal))
# file_list_neg = organize_DC_neg(list_file_time_order(file_list_signal))

# l1 = len(file_list_pos)
# l2 = len(file_list_neg)
# lmin = np.min([l1,l2])

# file_list_pos = file_list_pos[:lmin]
# file_list_neg = file_list_neg[:lmin]

Jx = Jnoise(file_list_noise, 0)

drive_t, drive2_t = get_drive(file_list_charge)

xph, dph, d2ph = getdata_x_d(file_list_charge[0])
phasebin = 0*getphase(drive_t, xph, Fs)

print len(file_list_signal)

# # corr, corr2 = do_corr(file_list_signal, 0, Jx)
corr, corr2 = do_corr_trigger(file_list_signal, phasebin, Jx, drive_t, drive2_t)


# corr_pos, bpos = do_corr_trigger(file_list_pos, phasebin, Jx, drive_t, drive2_t)
# corr_neg, bneg = do_corr_trigger(file_list_neg, phasebin, Jx, drive_t, drive2_t)

print np.mean(corr)
print np.mean(corr2)


print "  "
print "e#/nucleon at w", ((np.real(np.mean(corr))/conversion41)*(Vpp/Vmeasurement_pp))/(Nucleons)
print "e#/nucleon at 2w", ((np.real(np.mean(corr2))/conversion82)*(Vpp/Vmeasurement_pp))/(Nucleons)


# print 0.5*((np.real(np.mean(corr_pos) + np.mean(corr_neg))/conversion41)*(Vpp/Vmeasurement_pp))/(Nucleons)

# a = list_corr(corr_pos,corr_neg)

# print ((np.mean(a)/conversion41)*(Vpp/Vmeasurement_pp))/(Nucleons)


# plt.figure()
# plt.plot(((np.real(corr_pos)/conversion41)*(Vpp/Vmeasurement_pp))/(Nucleons), 'ro')
# plt.plot(((np.real(corr_neg)/conversion41)*(Vpp/Vmeasurement_pp))/(Nucleons), 'bo')
# plt.plot(((np.real((corr_pos+corr_neg)/2.)/conversion41)*(Vpp/Vmeasurement_pp))/(Nucleons), 'go')
# plt.plot(((a/conversion41)*(Vpp/Vmeasurement_pp))/(Nucleons), 'ko')
plt.grid()


# plt.figure()
# plt.plot(((np.real(bpos + bneg)/conversion82)*(Vpp/Vmeasurement_pp))/(Nucleons), 'ro')
# plt.grid()
# # plt.show()

# aaa = ((np.real((corr_pos+corr_neg)/2.)/conversion41)*(Vpp/Vmeasurement_pp))/(Nucleons)

aaa = ((np.real((corr)/1.)/conversion41)*(Vpp/Vmeasurement_pp))/(Nucleons)

def gauss(x,a,b,c):
    g = c*np.exp(-0.5*((x-a)/b)**2)
    return g

h,b = np.histogram(aaa, bins = 15)

bc = np.diff(b)/2 + b[:-1]

p0 = [0.1*10**-19, 5.*10**-19, 2]
try:
    popt, pcov = curve_fit(gauss, bc, h, p0)
except:
    popt = p0

xxx = np.linspace(bc[0],bc[-1], 100)

print "result from charge fit in e#"
print popt
print np.sqrt(pcov[0,0])


plt.figure()
plt.errorbar(bc, h, yerr = np.sqrt(h), fmt = 'ko')
plt.plot(xxx, gauss(xxx,*popt))
plt.xlabel("Electron Number")
plt.grid()

plt.figure()
plt.plot(aaa, 'ro')





#############################################
#############################################

# from electron number to Force:

# conversion41_force = (np.mean(corr)/(electron*Vpp*0.5/d)) # for the calibration file. the factor 0.5 comes from vpp to vamp
# print conversion41_force

conversion41_force = 1.19575706935e+16

# print "force in Newtons"
# print np.real(np.mean(corr))/conversion41_force

force_points = ((np.real((corr)/1.)/conversion41_force))

h1,b1 = np.histogram(force_points, bins = 15)

bc1 = np.diff(b1)/2 + b1[:-1]

p01 = [-1.*10**-19, 5.*10**-18, 2]
try:
    popt1, pcov1 = curve_fit(gauss, bc1, h1, p01)
except:
    popt1 = p01

xxx1 = np.linspace(bc1[0],bc1[-1], 100)

print "result from force fit Newtons"
print popt1
print np.sqrt(pcov1[0,0])


plt.figure()
plt.errorbar(bc1, h1, yerr = np.sqrt(h1), fmt = 'ko')
plt.plot(xxx1, gauss(xxx1,*popt1))
plt.xlabel("Force [N]")
plt.grid()


# from electron number to acceleration:

mass = (1.18*10**-11)/3.6 #kg


acc_points = ((np.real((corr)/1.)/conversion41_force))/mass

h2,b2 = np.histogram(acc_points, bins = 15)

bc2 = np.diff(b2)/2 + b2[:-1]

p02 = [0.1*10**-5, 5.*10**-5, 2]
try:
    popt2, pcov2 = curve_fit(gauss, bc2, h2, p02)
except:
    popt2 = p02

xxx2 = np.linspace(bc2[0],bc2[-1], 100)

print "result from acceleration fit SI units"
print popt2
print np.sqrt(pcov2[0,0])


plt.figure()
plt.errorbar(bc2, h2, yerr = np.sqrt(h2), fmt = 'ko')
plt.plot(xxx2, gauss(xxx2,*popt2))
plt.xlabel("acceleration [m/s^2]")
plt.grid()

plt.show()
