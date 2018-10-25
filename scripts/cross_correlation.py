import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob

fname = r"XGain_20_PID_0.h5"
path = r"C:\data\20180426_feedback_before_chamber\feedback_X"

NFFT = 2**14

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

	xycsd, freqs = matplotlib.mlab.csd(dat[:, bu.xi]-numpy.mean(dat[:, bu.xi]), dat[:, bu.yi]-numpy.mean(dat[:, bu.yi]), Fs = Fs, NFFT = NFFT)

        xpsd, freqs = matplotlib.mlab.psd(dat[:, bu.xi]-numpy.mean(dat[:, bu.xi]), Fs = Fs, NFFT = NFFT) 
        ypsd, freqs = matplotlib.mlab.psd(dat[:, bu.yi]-numpy.mean(dat[:, bu.yi]), Fs = Fs, NFFT = NFFT)
	
	return [freqs, xycsd, xpsd, ypsd]

freqs, xycsd, xpsd, ypsd = getdata(os.path.join(path, fname))

print np.sum(xycsd)/np.sqrt(np.sum(ypsd)*np.sum(xpsd))

plt.figure()
plt.loglog(freqs, np.abs(xycsd), label = "abs of csd")
plt.loglog(freqs, xpsd, label = "xpsd")
plt.loglog(freqs, ypsd, label = "ypsd")
plt.legend()
plt.show()
