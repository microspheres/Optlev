import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
import scipy.optimize as opt

name = r"2.4mbar_zcool.h5"
path = r"C:\data\20190115\15um\2"

comp_file = "2.0e-7mbar_xyzcool.h5"

f_start = 40.
f_end = 400.

NFFT = 2**14

kb = 1.38*10**-23

mass = 2.*2.3*10**-26

vis = 18.54*10**-6

rho = 1800

R = 7.5*10**-6

M = (4./3.)*np.pi*(R**3)*rho

press = 360.

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

        x = dat[:, bu.xi]-numpy.mean(dat[:, bu.xi])

        y = dat[:, bu.yi]-numpy.mean(dat[:, bu.yi])
        
	return [freqs, xpsd, ypsd, x, y, Fs]

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
    s1 = 2.*kb*temp*(gamma*(w0**2))
    s2 = 1.*M*(w0**2)*((w0**2 - w**2)**2 + (gamma*w)**2)
    s = np.sqrt(s1/s2)
    return A*s

gamma = Gamma(vis, press, temp, mass, R, M)

fit_points = np.logical_and(data[0] > f_start, data[0] < f_end )

f = np.arange(f_start, f_end, 1)

px, cx = opt.curve_fit(psd, data[0][fit_points], np.sqrt(data[1][fit_points]), p0 = [1e6, 100, gamma] )

py, cy = opt.curve_fit(psd, data[0][fit_points], np.sqrt(data[2][fit_points]), p0 = [1e6, 100, gamma] )


comp = getdata(os.path.join(path, comp_file))




fig = plt.figure()
plt.subplot(2, 1, 1)
plt.loglog(data[0], np.sqrt(data[1])/px[0],label="x")
plt.loglog(f, psd(f, *px)/px[0])

plt.loglog(comp[0], np.sqrt(comp[1])/px[0],label="comparison_x")
        
plt.grid()
plt.ylabel("$m/ \sqrt{Hz}$")
plt.legend(loc=3)
plt.subplot(2, 1, 2)
plt.loglog(data[0], np.sqrt(data[2])/py[0], label = "y")
plt.loglog(f, psd(f, *py)/py[0])

plt.loglog(comp[0], np.sqrt(comp[2])/py[0],label="comparison_y")
        
plt.xlabel("Frequency[Hz]")
plt.ylabel("$m/ \sqrt{Hz}$")
plt.legend(loc=3)
plt.grid()

filter = True
if filter:
    from scipy import signal
    Fs = comp[5]
    def butter_bandpass(L, H, fs, order):
        nyq = 0.5 * fs
        nH = H / nyq
        nL = L / nyq
        b, a = signal.butter(order, [nL, nH], btype='bandpass', analog=False)
        return b, a

    def butter_bandpass_filter(data, L, H, fs, order):
        b, a = butter_bandpass(L, H, fs, order=order)
        y = signal.filtfilt(b, a, data)
        return y

    compfx = butter_bandpass_filter(comp[3], 70., 90., Fs, 1)



stdx = np.std(comp[3])/px[0]
stdy = np.std(comp[4])/py[0]

T = numpy.linspace(0,1e4,1e5)
dT = T[1]-T[0]
vx = numpy.gradient(comp[3]/px[0], dT)
vy = numpy.gradient(comp[4]/py[0], dT)

stdvx = np.std(vx)
stdvy = np.std(vy)

hbar = 1.0545718*10**(-34)

nx = M*stdx*stdvx/(hbar) - 0.5
ny = M*stdy*stdvy/(hbar) - 0.5

print stdvx
print stdvy

print "nx = ", nx
print "ny = ", ny

print ((np.std(compfx)/px[0])**2)*(2.*M*(2.*np.pi*abs(px[1])))/(2.*hbar)
print np.std(compfx)/px[0]

plt.show()

plt.figure()
plt.plot(comp[3])
plt.plot(compfx)
plt.show()
