import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob

CPf = r"X_PM167mW_Constant_power.h5"
CCf = r"X_PM152mW_Constant_current.h5"
path = r"C:\data\PM_test"

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
        
	return [freqs, xpsd, dat[:, bu.xi]]

CP = getdata(os.path.join(path, CPf))
CC = getdata(os.path.join(path, CCf))


fig = plt.figure()
plt.loglog(CP[0], np.sqrt(CP[1]),label="CP")
plt.loglog(CC[0], np.sqrt(CC[1]),label="CC")
plt.legend()
plt.ylabel("V/rtHz")
plt.xlabel("Frequency[Hz]")
plt.show()
