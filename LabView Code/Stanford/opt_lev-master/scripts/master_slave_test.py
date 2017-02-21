import numpy, h5py
import matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp


fname_m = r"_2500mV_10Hztest_20.h5"
fname_s = r"_2500mV_10Hztest_2.h5"
path_m = r"D:\Data\test"
path_s = r"D:\Data\test"
d2plt = 1

Fs = 5e3  ## this is ignored with HDF5 files
NFFT = 2**11
def getdata(fname):
	print "Opening file: ", fname
	## guess at file type from extension
	_, fext = os.path.splitext( fname )
	if( fext == ".h5"):
		f = h5py.File(fname,'r')
		dset = f['beads/data/pos_data']
		dat = numpy.transpose(dset)
		max_volt = dset.attrs['max_volt']
		nbit = dset.attrs['nbit']
		Fs = dset.attrs['Fsamp']
		
		dat = 1.0*dat*max_volt/nbit

	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5] )

	xpsd, freqs = matplotlib.mlab.psd(dat[:, 0], Fs = Fs, NFFT = NFFT) 
	ypsd, freqs = matplotlib.mlab.psd(dat[:, 0], Fs = Fs, NFFT = NFFT)
        zpsd, freqs = matplotlib.mlab.psd(dat[:, 0], Fs = Fs, NFFT = NFFT)
	norm = numpy.median(dat[:, 0])
	return [freqs, xpsd, ypsd, dat, zpsd]

data_m = getdata(os.path.join(path_m, fname_m))
data_s = getdata(os.path.join(path_s, fname_s))

plt.plot(data_m[3][:, -1])
plt.plot(data_s[3][:, -1])
plt.show()
        



