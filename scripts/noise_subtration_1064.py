import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
import scipy.optimize as opt
from scipy.signal import butter, lfilter, filtfilt


NFFT = 2**14


path_list = [r"C:\data\201904011\15um\beforeFB\6\temp\samegain\deflection4",]


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

        xfft = np.fft.rfft(x, n = NFFT)
        xbfft = np.fft.rfft(xb, n = NFFT)
        freqs = Fs*np.fft.rfftfreq(n = NFFT)

        freq, csd = sp.csd(xb, x, fs = Fs, nperseg = NFFT)
        freq, xbpsd = sp.csd(xb, xb, fs = Fs, nperseg = NFFT)
        
        freq, coherence = sp.coherence(x, xb, fs = Fs, nperseg = NFFT)

        proj = (xbfft/xbpsd)*csd

        x_subfft = xfft - proj
        
        #plt.figure()
        # plt.loglog(freqs, np.abs(xfft), label = "x")
        # plt.loglog(freqs, np.abs(xbfft), label = "xb")
        # plt.loglog(freqs, np.abs(x_subfft), label = "x_sub")
        # plt.xlim(1, 2000)
        # plt.legend()
        #plt.show()

        return [freq, xfft, xbfft, x_subfft, coherence]


    
def get_files_path(path):
    file_list = glob.glob(path+"\*.h5")
    return file_list


def get_data_path(path): # PSD output is unit square, V**2/Hz : it assumes that within the folder, Dgx is the same.
    info = getdata(get_files_path(path)[0])
    freq = info[0]
    Xfft = np.zeros(len(freq), dtype=complex)
    Xbfft = np.zeros(len(freq), dtype=complex)
    X_subfft = np.zeros(len(freq), dtype=complex)
    Coh = np.zeros(len(freq))
    aux = get_files_path(path)
    for i in aux:
        print i
        a = getdata(i)
        Xfft += a[1]
        Xbfft += a[2]
        X_subfft += a[3]
        Coh += a[4]
        
    Xfft = Xfft/len(aux)
    Xbfft = Xbfft/len(aux)
    X_subfft = X_subfft/len(aux)
    Coh = Coh/len(aux)
    return [freq, Xfft, Xbfft, X_subfft, Coh]



a = get_data_path(path_list[0])

plt.figure()
# plt.loglog(a[0], np.sqrt(np.abs(a[1])), label = "X")
# plt.loglog(a[0], np.sqrt(np.abs(a[2])), label = "Xb")
# plt.loglog(a[0], np.sqrt(np.abs(a[3])), label = "Xsub")

plt.loglog(a[0], np.sqrt(np.abs(a[4])), label = "Coherence")

plt.legend()    
plt.show()
