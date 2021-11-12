import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob

FPGAout = r"X_PM_Y_SYNTH.h5"
path = r"C:\data\AOM"

NFFT = 2**17

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
        
	return [freqs, xpsd, ypsd, dat[:, bu.xi], dat[:, bu.yi]]

a = getdata(os.path.join(path, FPGAout))
t = range(len(a[3]))

print a[3]


fig = plt.figure()
plt.plot(a[3], a[4],label="PM vs synth")
plt.legend()
plt.xlabel("PM [V]")
plt.ylabel("synth [V]")
plt.show()
