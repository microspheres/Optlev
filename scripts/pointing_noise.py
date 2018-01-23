import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob

	 
path_pointing_drive = r"C:\data\20171031\bead1_15um_QWP_NS\X_modulation"

path_bf_low_pressure = r"C:\data\20171031\bead1_15um_QWP_NS\meas\DC_no_AC2"

path_save = r"C:\data\acceleration_paper\from_dates\20171031bead1_15um_QWP_NS\pointing_noise_low_pressure\psd"

file_list_psd = glob.glob(path_bf_low_pressure+"\*.h5")
file_list_drive = glob.glob(path_pointing_drive+"\*.h5")


NFFT = 2**19

startfile = 500
endfile = 600

def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist

file_list_psd = list_file_time_order(file_list_psd)

file_list_psd = file_list_psd[startfile:endfile]


V_to_g = 0.0038 # comes from the sensitivity_plot2.py


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
        before_chan, freqs = matplotlib.mlab.psd(dat[:, bu.xl]-numpy.mean(dat[:, bu.xl]), Fs = Fs, NFFT = NFFT)
        aux = np.argmax(before_chan[3:-1])
        freq_drive = freqs[aux]
	return [freqs, xpsd, before_chan, freq_drive, aux]

def unite_psd(file_list): # only for the signal before chamber
    freqs = np.array(getdata(file_list[0])[0])
    before = np.zeros(len(freqs))
    for file in file_list:
       a = getdata(file)
       before += np.array(a[2])
    return [freqs, before/len(file_list)]


def signal_bf_vs_signal_sphere(file_list): # converts signal before the chamber to signal in the X sensor
    a = 2
    freq= getdata(file_list[0])[0]
    peak = getdata(file_list[0])[4]
    dfreq = freq[peak] - freq[peak - 1]
    xpsd = getdata(file_list[0])[1]
    before = getdata(file_list[0])[2]
    v2_xpsd = np.sum(xpsd[peak - a:peak + a])*dfreq
    v2_before = np.sum(before[peak - a:peak + a])*dfreq
    return 1./(1.0*v2_before/v2_xpsd)

conv = np.sqrt(signal_bf_vs_signal_sphere(file_list_drive))

A = unite_psd(file_list_psd)

psd_noise = conv*np.sqrt(A[1])*V_to_g
# print 20*conv*V_to_g/(2*np.pi*220)**2
freq = A[0]

# power_factor = 1.2 # different sphere used (ratio of powers as the signal is prop to the 1064 power)
# Xsensor_factor = 1.3 # different sphere used (ration of the sphere area as the signal is prop to that)

# psd_noise = psd_noise/(power_factor*Xsensor_factor)



def plot_sensitivity_g(): # check that the sensor gain for calibration is the same for the measurement!
    # A = unite_psd(file_list_psd)
    plt.figure()
    plt.ylabel('acceleration [ug]/$\sqrt{Hz}$')
    plt.xlabel('Freq[Hz]')
    plt.loglog(freq, 20*psd_noise*1e6) # 20 is beacause of the gain
    plt.grid()
    np.savetxt(os.path.join(path_save,'freq_vs_g.txt'), (freq, 20*psd_noise*1e6))
    return



plot_sensitivity_g()
plt.show()
