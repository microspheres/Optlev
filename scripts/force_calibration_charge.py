#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 09:09:16 2017

@author: fernandomonteiro
"""

import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import os, re, time, glob
import bead_util as bu

electric_charge = 1.602e-19
distance = 0.002 #mm


path = r"/Volumes/FERNANDO/lab/20170511/bead2_15um_QWP/charge6"

make_plot_vs_time = True
		 

Fs = 10e3  ## this is ignored with HDF5 files
NFFT = 2**13

def getdata(fname):
        print "Opening file: ", fname
	## guess at file type from extension
	_, fext = os.path.splitext( fname )
	if( fext == ".h5"):
		f = h5py.File(fname,'r')
		dset = f['beads/data/pos_data']
		dat = numpy.transpose(dset)
		#max_volt = dset.attrs['max_volt']
		#nbit = dset.attrs['nbit']
		Fs = dset.attrs['Fsamp']
		
		#dat = 1.0*dat*max_volt/nbit
                dat = dat * 10./(2**15 - 1)
                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )

	xpsd, freqs = matplotlib.mlab.psd(dat[:, 0]-numpy.mean(dat[:, 0]), Fs = Fs, NFFT = NFFT) 
        drive, freqs = matplotlib.mlab.psd(dat[:, 5]-numpy.mean(dat[:, 5]), Fs = Fs, NFFT = NFFT)
        # zpsd, freqs = matplotlib.mlab.psd(dat[:, 2]-numpy.mean(dat[:, 2]), Fs = Fs, NFFT = NFFT)

	# norm = numpy.median(dat[:, 2])
        #for h in [xpsd, ypsd, zpsd]:
        #        h /= numpy.median(dat[:,2])**2
	# return [freqs, xpsd, ypsd, dat, zpsd]
        return [freqs, xpsd, drive]



def list_file_time_order(p):
    filelist = glob.glob(os.path.join(p,"*.h5"))
    filelist.sort(key=os.path.getmtime)
    return filelist

#def get_max_bin(A,B):
#    a = np.max(A)
#    b = 0
#    for i in np.arange(len(A)):
#        if a == A[i]:
#            b = B[i]
#    return 1.0*b

def get_max_bin(limI,limS,p): # output in V/sqrt(Hz)
    freq = np.zeros(NFFT/2 + 1)
    x = np.zeros(NFFT/2 + 1)
    dx = np.zeros(NFFT/2 + 1)
    for file in list_file_time_order(p)[limI:limS]:
        a = getdata(file)
        freq = a[0]
        x += np.sqrt(a[1])
        dx += np.sqrt(a[2])
    a1 = np.max(dx)
    b1 = 0
    for i in np.arange(len(dx)):
        if a1 == dx[i]:
            b1 = x[i]
    return 1.0*b1/len(list_file_time_order(p)[limI:limS])

def get_field_AC(p): # all files MUST have the same fields value!!!
    trek_factor = 200.0
    Vpp_to_Vamp = 0.5
    filelist = glob.glob(os.path.join(p,"*.h5"))
    path_file1 = filelist[0]
    i = path_file1.rfind("synth")+5
    j = path_file1.rfind("mV")
    k = path_file1.rfind("mV",0,j)
    file1 = path_file1[i:k]
    voltage = float(file1)/1000.
    return Vpp_to_Vamp*trek_factor*(voltage)/distance

def convert_voltage_to_force(limI_1e, limS_1e, limI_0e, limS_0e, p): #gives the convertion factor from V/sqrtHz to Newtons
    force = electric_charge*get_field_AC(path)
    delta_v = get_max_bin(limI_1e,limS_1e,p) - get_max_bin(limI_0e,limS_0e,p)
    return 1.0*force/delta_v

def PSD_Newton_1s(position, p, conversion):
    freq = np.zeros(NFFT/2 + 1)
    x = np.zeros(NFFT/2 + 1)
    
    a = getdata(list_file_time_order(p)[position])
    freq = a[0]
    x += np.sqrt(a[1])
    
    new_x = conversion*x
    plt.figure()
    plt.loglog(freq,new_x)
    plt.grid()
    plt.show()

conversion = convert_voltage_to_force(0, 57, 100, 157, path)
PSD_Newton_1s(100, path, conversion)

# adding the AC voltage vs. frequency amplitude here
# also, path here is
# "/data/20170511/bead2_15um_QWP/new_sensor_feedback/charge7_piezo_56.9_74.9_75.4"
def getACAmplitudeGraphs(file_list):
    """output AC voltages and corresponding amplitudes at both omega and 2 omega"""
    N = len(file_list)
    constant = conversion/N
    x = {} # input only numpy arrays as values
    dx = {} # input only numpy arrays as values
    voltageCount = {} # input integers that count how many times an AC voltage value has shown up
    for index in range(N):
        f = file_list[index]
        a = getdata(f)
        i = f.rfind("synth")+5
        j = f.rfind("mV")
        k = f.rfind("mV",0,j)
        ACvoltage = float(f[i:k])/1000.
        if ACvoltage in x:
            voltageCount[ACvoltage] += 1
            x[ACvoltage] += np.sqrt(a[1])
            dx[ACvoltage] += np.sqrt(a[2])
        else:
            voltageCount[ACvoltage] = 1
            x[ACvoltage] = np.sqrt(a[1])
            dx[ACvoltage] = np.sqrt(a[2])
    ACvoltages = x.keys()
    N1 = len(ACvoltages)
    omegaAmplitudes = range(N1)
    twoOmegaAmplitudes = range(N1)
    """Now insert the amplitude for the requisite frequencies"""
    for index in range(N1):
        i = np.argmax(dx[ACvoltages[index]])
        omegaAmplitudes[index] = constant*x[i]
        twoOmegaAmplitudes[index] = constant*x[2*i]
    return ACvoltages, omegaAmplitudes, twoOmegaAmplitudes
