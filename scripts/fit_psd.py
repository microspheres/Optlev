import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
import scipy.optimize as opt

name = r"4mbar_XY.h5"
path = r"C:\data\20190108\15um"

f_start = 50.
f_end = 300.

NFFT = 2**12

kb = 1.38*10**-23

mass = 2.*2.3*10**-26

vis = 18.54*10**-6

rho = 1800

R = 7.5*10**-6

M = (4./3.)*np.pi*(R**3)*rho

press = 440.

temp = 300

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
	ypsd, freqs = matplotlib.mlab.psd(dat[:, bu.yi]-numpy.mean(dat[:, bu.yi]), Fs = Fs, NFFT = NFFT)
        
	return [freqs, xpsd, ypsd]

data = getdata(os.path.join(path, name))

####################

def mean_free_path(vis, press, temp, mass):
    L1 = vis/press
    L2 = np.sqrt( np.pi*kb*temp/(2*mass) )
    return L1*L2

def Kn(vis, press, temp, mass, R):
    L = mean_free_path(vis, press, temp, mass)
    return L/R

def Gamma(vis, press, temp, mass, R, M):
    A = (6.0*np.pi*vis*R/M)
    B = 0.619/(0.619 + Kn(vis, press, temp, mass, R))
    C = (1. + 0.31*Kn(vis, press, temp, mass, R)/(0.785 + 1.152*Kn(vis, press, temp, mass, R)) )
    return A*B*C

def psd(f, A, f0, gamma):
    w0 = 2.*np.pi*f0
    w = 2.*np.pi*f
    s1 = 2.*kb*temp*(gamma*(w0)**2)
    s2 = 1.*M*(w0**2)*((w0**2 - w**2)**2 + (gamma*w0)**2)
    s = np.sqrt(s1/s2)
    return A*s

gamma = Gamma(vis, press, temp, mass, R, M)

fit_points = np.logical_and(data[0] > f_start, data[0] < f_end )

f = np.arange(f_start, f_end, 1)

px, cx = opt.curve_fit(psd, data[0][fit_points], np.sqrt(data[1][fit_points]), p0 = [1e6, 100, gamma] )

py, cy = opt.curve_fit(psd, data[0][fit_points], np.sqrt(data[2][fit_points]), p0 = [1e6, 100, gamma] )




fig = plt.figure()
plt.subplot(2, 1, 1)
plt.loglog(data[0], np.sqrt(data[1])/px[0],label="x")
plt.loglog(f, psd(f, *px)/px[0])

plt.ylabel("m/rtHz")
plt.legend(loc=3)
plt.subplot(2, 1, 2)
plt.loglog(data[0], np.sqrt(data[2])/py[0])
plt.loglog(f, psd(f, *py)/py[0])
plt.xlabel("Frequency[Hz]")

plt.show()

print px
print py
