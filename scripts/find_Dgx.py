import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
import scipy.optimize as opt

refname = r"1mbar_zcool.h5"
fname0 = r"1mbar_xyzcool.h5"
path = r"C:\data\20191122\10um\2\1mbar"

		 
NFFT = 2**15

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
                pid = dset.attrs['PID']
		
		#dat = 1.0*dat*max_volt/nbit
                dat = dat * 10./(2**15 - 1)
                Press = dset.attrs['pressures'][0]
                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )

	xpsd, freqs = matplotlib.mlab.psd(dat[:, bu.xi]-numpy.mean(dat[:, bu.xi]), Fs = Fs, NFFT = NFFT)
        
        return [freqs, xpsd, pid[0]]

data0 = getdata(os.path.join(path, refname))
data1 = getdata(os.path.join(path, fname0))

def harmonic(f, f0, A, gamma):
    w0 = 2.*np.pi*np.abs(f0)
    w = 2.*np.pi*f
    gamma = 2.0*np.pi*gamma

    a1 = 1.*np.abs(A*gamma)
    a3 = 1.*(w0**2 - w**2)**2 + (w*gamma)**2

    s = 1.*a1/a3

    return s



def harmonic_feedback(f, f0, A, gamma, gain, Dg):

    w0 = 2.*np.pi*np.abs(f0)
    w = 2.*np.pi*f
    gamma = 2.0*np.pi*gamma

    H = -1j*w

    a1 = 1.*np.abs(A*gamma)
    a3 = 1.*(w0**2 - w**2)**2 + (w*gamma)**2 - 2.*gamma*w*(w0**2)*(gain*Dg)*np.imag(H) + (w0**4)*((gain*Dg)**2)*(np.abs(H)**2)

    s = 1.*a1/a3

    return s



popt, pcov = opt.curve_fit(harmonic, data0[0][100:300], data0[1][100:300]/(2.*np.pi))
print popt

def test(f, f0, gain):
    return harmonic_feedback(f, f0, np.abs(popt[1]), popt[2], gain,  np.abs(data1[2]))

poptf, pcovf = opt.curve_fit(test, data1[0][100:300], data1[1][100:300]/(2.*np.pi))
print poptf

print "gain = ", poptf[1]

plt.figure()
plt.loglog(data0[0][100:300], data0[1][100:300])
plt.loglog(data0[0], (2.*np.pi)*harmonic(data0[0], *popt))
plt.loglog(data1[0][100:300], data1[1][100:300])
plt.loglog(data1[0], (2.*np.pi)*test(data1[0], *poptf))
plt.show()

