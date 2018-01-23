import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob

# 15um
# path_charge = r"C:\data\20170726\bead8_15um_QWP\steps\calibration_1positive"
# file_list_charge = glob.glob(path_charge+"\*.h5")

# path_psd = r"C:\data\20170726\bead8_15um_QWP"
# file1 = r"1_6E-6mbar_xyzcool_G5_att.h5"

# 23 um		 
path_charge = r"C:\data\20171002\bead2_23um_QWP_NS\calibration_1p\1"
file_list_charge = glob.glob(path_charge+"\*.h5")

path_psd = r"C:\data\20171002\bead2_23um_QWP_NS\meas\DC_no_AC2"
file1 = r"auto_xyzcool_G100_att_0.h5"

mass = 1.0*10**-11 #kg

Fs = 10e3  ## this is ignored with HDF5 files
NFFT = 2**19

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

	xpsd, freqs = matplotlib.mlab.psd(dat[:, bu.xi]-numpy.mean(dat[:, bu.xi]), Fs = Fs, NFFT = NFFT)
        drivepsd, freqs = matplotlib.mlab.psd(dat[:, bu.drive]-numpy.mean(dat[:, bu.drive]), Fs = Fs, NFFT = NFFT)
        aux = np.argmax(drivepsd)
        freq_drive = freqs[aux]
	return [freqs, xpsd, drivepsd, freq_drive, aux]

def unite_psd(file_list): # for charge calibration
    freqs = np.array(getdata(file_list[0])[0])
    X = np.zeros(len(freqs))
    for file in file_list:
       a = getdata(file)
       X += np.array(a[1])
    return [freqs, X/len(file_list)]

def Voltsquare_at_peak(file_list): # area of the peak
    a = 2
    freq, xpsd = unite_psd(file_list)
    peak = getdata(file_list[0])[4]
    dfreq = freq[peak] - freq[peak - 1]
    v2 = np.sum(xpsd[peak - a:peak + a])*dfreq
    return v2

def v_to_newton(file_list):
    distance = 0.001 #m
    v = 200.0*0.3 # volts
    E = 1.0*v/distance
    charge = 1.602*10**(-19) # SI units
    force = charge*E
    conversion = force/np.sqrt(Voltsquare_at_peak(file_list))
    return conversion

def v_to_g(file_list):
    distance = 0.001 #m
    v = 200.0*0.3 # volts
    E = 1.0*v/distance
    charge = 1.602*10**(-19) # SI units
    force = charge*E
    a = force/mass
    conversion = a/np.sqrt(Voltsquare_at_peak(file_list))
    return conversion

def v_to_electron(file_list):
    conversion = 1/np.sqrt(Voltsquare_at_peak(file_list))
    N = 1.0*(10**15)*(3.1)
    ratio_voltage = 0.1/20.0
    return (conversion/N)*ratio_voltage

def plot_sensitivity_force(file0, path, file_list_charge):
    A = getdata((os.path.join(path, file0)))
    c = v_to_newton(file_list_charge)
    cn = c*20.0 # this number is due to different sensor gain 100/5 = 20
    plt.figure()
    plt.ylabel('Force[N]')
    plt.xlabel('Freq[Hz]')
    plt.loglog(A[0], c*np.sqrt(A[1]))
    plt.grid()
    return

def plot_sensitivity_acceleration(file0, path, file_list_charge):
    A = getdata((os.path.join(path, file0)))
    c = v_to_g(file_list_charge)
    c = c/9.8
    cn = c*20.0 # this number is due to different sensor gain 100/5 = 20
    plt.figure()
    plt.ylabel('Acceleration [g]')
    plt.xlabel('Freq[Hz]')
    plt.loglog(A[0], c*np.sqrt(A[1]))
    plt.grid()
    return

def plot_sensitivity_electron(file0, path, file_list_charge):
    A = getdata((os.path.join(path, file0)))
    c = v_to_electron(file_list_charge)
    cn = c*20.0 # this number is due to different sensor gain 100/5 = 20
    plt.figure()
    plt.ylabel('electron number')
    plt.xlabel('Freq[Hz]')
    plt.loglog(A[0], c*np.sqrt(A[1]))
    plt.grid()
    return

plot_sensitivity_force(file1, path_psd, file_list_charge)
plot_sensitivity_electron(file1, path_psd, file_list_charge)
plt.show()

# A1 = getdata((os.path.join(path_psd, file1)))
# c =  v_to_newton(file_list_charge)
# ce = v_to_electron(file_list_charge)

# gain_factor = 20.0
# cg = ce*gain_factor
# cn = c*gain_factor
# D = unite_psd(file_list_charge)


# peak = getdata(file_list_charge[0])[4]
# dfreqs = getdata(file_list_charge[0])[0][peak] - getdata(file_list_charge[0])[0][peak-1]

# dfreqs1 = A1[0][100] - A1[0][99]


# plt.figure()
# plt.loglog(A1[0], cn*np.sqrt(A1[1]))
# plt.loglog(D[0], c*np.sqrt(D[1]))
# plt.show()

# print np.sum(ce*np.sqrt(D[1])[peak-2:peak+2])*((10**15)/27.0)*(20.0/0.1)*np.sqrt(dfreqs)
