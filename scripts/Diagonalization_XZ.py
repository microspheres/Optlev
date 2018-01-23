import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
from scipy.signal import butter, lfilter, filtfilt
from scipy import signal




file_Zmod = r"1mbar_xyzcool_G100_att_Zmod.h5"

file_AC = r"1mbar_xyzcool_G100_att_ACx_synth500mV47Hz0mVdc.h5"

file_for_ABC = r"1mbar_xyzcool_G100_att.h5"

path = r"C:\data\20170919\bead2_15um_QWP_NS\diag"


file_AC = os.path.join(path, file_AC)
file_Zmod = os.path.join(path, file_Zmod)
file_for_ABC = os.path.join(path, file_for_ABC)



NFFT = 2**13

Fs = 10000

butterp = 1

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



def getdata_time_stream(fname):
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

        x = dat[:, bu.xi]-numpy.mean(dat[:, bu.xi])
        z = dat[:, bu.zi]-numpy.mean(dat[:, bu.zi])
        y = dat[:, bu.yi]-numpy.mean(dat[:, bu.yi])

	return [x, y, z]


def PSD(x,y,z):
    xpsd, freqs = matplotlib.mlab.psd(x-numpy.mean(x), Fs = Fs, NFFT = NFFT)
    ypsd, freqs = matplotlib.mlab.psd(y-numpy.mean(y), Fs = Fs, NFFT = NFFT)
    zpsd, freqs = matplotlib.mlab.psd(z-numpy.mean(z), Fs = Fs, NFFT = NFFT)
    return [freqs, xpsd, ypsd, zpsd]


def norm(x,y,z):
    x = x/np.sqrt(x*x)
    y = y/np.sqrt(y*y)
    z = z/np.sqrt(z*z)
    return [x,y,z]


def coef(file_list):
    x,y,z = getdata_time_stream(file_list)
    
    x,y,z = butter_bandpass_filter([x,y,z], 49, 51, Fs, 2)
    
    
    zort = z
    cyz = np.sum(y*zort)/np.sum(zort*zort)
    yort = y - cyz*zort
    cxz = np.sum(x*zort)/np.sum(zort*zort)
    cxy = np.sum(x*yort)/np.sum(yort*yort)
    xort = x - cxz*zort - cxy*yort
    
    return [cyz, cxz, cxy]

def orth(x, y, z, cyz, cxz, cxy):
    z = z
    y = y - cyz*z
    x = x - cxz*z - cxy*y
    return [x,y,z]


c1,c2,c3 = coef(file_Zmod)

x0,y0,z0 = getdata_time_stream(file_AC)


x,y,z = orth(x0,y0,z0, c1,c2,c3)

x = x

# rotate xy

# x,y = butter_bandpass_filter([x,y], 46.5, 47.5, Fs, 2)

def rotation2d(x,y,t):
    x = x*np.cos(t) - y*np.sin(t)
    y = x*np.sin(t) + y*np.cos(t)
    return [x,y]

plt.figure()
plt.plot(x,y,".",ms = 1)
    
x,y = rotation2d(x,y,-0.024)

plt.plot(x,y,".",ms = 1)
plt.show()

freq, xpsd,ypsd,zpsd = PSD(x,y,z)
freq, x0psd,y0psd,z0psd = PSD(x0,y0,z0)

plt.figure()
plt.subplot(3, 1, 1)
plt.loglog(freq, np.sqrt(x0psd))
plt.loglog(freq, np.sqrt(xpsd))
plt.subplot(3, 1, 2)
plt.loglog(freq, np.sqrt(y0psd))
plt.loglog(freq, np.sqrt(ypsd))
plt.subplot(3, 1, 3)
plt.loglog(freq, np.sqrt(z0psd))
plt.loglog(freq, np.sqrt(zpsd))
plt.show()

print c1,c2,c3
