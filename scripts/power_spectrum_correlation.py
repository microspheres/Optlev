import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob

name = r"LPmbar_xyzcool6_0.h5"
path = r"C:\data\20191119\10um\4\temp_x2\16"

NFFT = 2**16

def getdata_coherence(fname):
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

	xpsdin, freqs = matplotlib.mlab.psd(dat[:, bu.xi]-numpy.mean(dat[:, bu.xi]), Fs = Fs, NFFT = NFFT)
        xpsdout, freqs = matplotlib.mlab.psd(dat[:, 4]-numpy.mean(dat[:, 4]), Fs = Fs, NFFT = NFFT)
        
        f, Cxy = sp.coherence(dat[:, bu.xi]-numpy.mean(dat[:, bu.xi]), (dat[:, 4]-numpy.mean(dat[:, 4])), Fs, nperseg = NFFT)

        new_xpsdout = xpsdout*Cxy

        return [freqs, xpsdin, xpsdout, new_xpsdout]

data = os.path.join(path, name)

data = getdata_coherence(data)

freq = data[0]

plt.figure()
plt.loglog(freq, np.sqrt(data[1]), label = "in")
plt.loglog(freq, np.sqrt(data[2]), label = "out")
plt.loglog(freq, np.sqrt(data[3]), label = "out_corr")
plt.legend()
plt.grid()
plt.show()


        
def getdata_grand_Schmidt(fname):
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

	xpsdin, freqs = matplotlib.mlab.psd(dat[:, bu.xi]-numpy.mean(dat[:, bu.xi]), Fs = Fs, NFFT = NFFT)
        xpsdout, freqs = matplotlib.mlab.psd(dat[:, 4]-numpy.mean(dat[:, 4]), Fs = Fs, NFFT = NFFT)
        
        rfftin = np.fft.rfft(dat[:, bu.xi]-numpy.mean(dat[:, bu.xi]))
        rfftout = np.fft.rfft(dat[:, 4]-numpy.mean(dat[:, 4]))
        f = np.fft.rfftfreq(dat[:, bu.xi].size, 1./Fs)

        fftin_filter = rfftin
        fftout_filter = rfftout - fftin_filter*( np.sum( rfftout*np.conj(fftin_filter) ) )/( np.sum( fftin_filter*np.conj(fftin_filter) ) )

        new_x_out = np.fft.irfft(fftout_filter)

        new_xpsdout, freqs = matplotlib.mlab.psd(new_x_out-numpy.mean(new_x_out), Fs = Fs, NFFT = NFFT)

        return [freqs, xpsdin, xpsdout, new_xpsdout]
