import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
import scipy.optimize as opt
from scipy.signal import butter, lfilter, filtfilt


NFFT = 2**15


path = r"C:\data\1064_selfnoise"


fname = r"Filter_ON.h5"
fname1 = r"Filter_OFF.h5"
fname2 = r"Filter_OFF_532_blocked.h5"

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
                PID = dset.attrs['PID']
                press = dset.attrs['pressures']
                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )

        x = dat[:, bu.xi]-numpy.mean(dat[:, bu.xi])

        xb = dat[:, 4] - numpy.mean(dat[:, 4])

        xpsd, freqs = matplotlib.mlab.psd(x, Fs = Fs, NFFT = NFFT)
        xbpsd, freqs = matplotlib.mlab.psd(xb, Fs = Fs, NFFT = NFFT) 

        return [freqs, xpsd, xbpsd]


    



a = getdata(os.path.join(path, fname))
b = getdata(os.path.join(path, fname1))
c = getdata(os.path.join(path, fname2))

plt.figure()

plt.loglog(a[0], np.sqrt(a[1]), label = "X_after_Filter_ON")
plt.loglog(a[0], np.sqrt(a[2]), label = "1064nm_before_chamber")

plt.loglog(b[0], np.sqrt(b[1]), label = "X_after_Filter_OFF")

plt.loglog(b[0], np.sqrt(c[1]), label = "X_after_532_blocked_filter_OFF")

plt.ylabel("PSD V/SqHz")
plt.xlabel("Freq [Hz]")
plt.xlim(1, 200)

plt.legend()    
plt.show()
