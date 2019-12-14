import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
import scipy.optimize as opt
from scipy.signal import butter, lfilter, filtfilt
def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist

folder_meas = r"C:\data\20191122\10um\2\charge_plot"

file_list_meas = glob.glob(folder_meas+"\*.h5")
file_list_meas = list_file_time_order(file_list_meas)

distance = 0.0021

drive_col = 3

NFFT = 2**13

def getdata(fname):
	# print "Opening file: ", fname
	## guess at file type from extension
	_, fext = os.path.splitext( fname )
	if( fext == ".h5"):
		f = h5py.File(fname,'r')
		dset = f['beads/data/pos_data']
		dat = numpy.transpose(dset)
		#max_volt = dset.attrs['max_volt']
		#nbit = dset.attrs['nbit']
		Fs = dset.attrs['Fsamp']
                PID = dset.attrs['PID']
		
		#dat = 1.0*dat*max_volt/nbit
                dat = dat * 10./(2**15 - 1)
                time = dset.attrs['Time']

                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )

	xin = dat[:, 0]-numpy.mean(dat[:, 0])
        xout = dat[:, 4]-numpy.mean(dat[:, 4])
        drive = dat[:, drive_col]-numpy.mean(dat[:, drive_col])

	return [xin, xout, drive, Fs, time]

def plot_charge(folder_meas): # return freq, freq_arg, amplitude of the drive and sphere x motion in the freq band.
    file_list = glob.glob(folder_meas+"\*.h5")
    file_list = list_file_time_order(file_list)

    i = file_list[0]
    a = getdata(i)

    drive = a[2]
    Fs = a[3]
    xin = a[0]
    xout = a[1]

    freq, drive_psd = sp.csd(drive, drive, Fs, nperseg=NFFT, scaling = "spectrum")
    
    f0arg = np.argmax(drive_psd)
    f0 = freq[f0arg]
    print f0

    E = np.sum(drive_psd[f0arg-1: f0arg+1])
    E = np.sqrt(2.*E)/distance

    # plt.figure()
    # plt.loglog(freq, drive_psd)
    # plt.loglog(freq[f0arg], drive_psd[f0arg], "ro")
    # plt.show()

    Charge = []
    Time = []
    
    for j in file_list:
        a = getdata(j)
        time = a[4]
        
        drive = a[2]
        Fs = a[3]
        xin = a[0]
        xout = a[1]
        
        freq, charge = sp.csd(xin, drive, Fs, nperseg=NFFT)
        freq, V2 = sp.csd(drive, drive, Fs, nperseg=NFFT)
        charge = np.real(charge[f0arg])/np.sqrt( np.abs( np.real( V2[f0arg] ) ))
        Charge.append(charge)
        Time.append(time)

    Time = np.array(Time)
    Time = Time - Time[0]

    index = np.where(Time > 61)[0][0]
    
    Charge = np.array(Charge)

    p1 = np.abs(np.mean(Charge[0:5])/9.)

    

    title = "E = " + str("%.1f" % E) + " V/m"
   
    plt.figure(figsize=(5.5,3))
    plt.rcParams.update({'font.size': 14})
    plt.plot(Time[index:] - Time[index:][0], Charge[index:]/p1, "r.")
    plt.xlabel("Time [S]")
    plt.ylabel("Charge [e$^-$]")
    plt.title(title)
    plt.grid(which='both')
    plt.tight_layout(pad = 0)
    return [f0, f0arg, drive, xin, xout, Fs, freq]

plot_charge(folder_meas)
plt.show()
