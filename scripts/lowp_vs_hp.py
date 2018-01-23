import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob

path_l = r"C:\data\20171031\bead1_15um_QWP_NS\meas\DC_no_AC2"

path_h = r"C:\data\20171031\bead1_15um_QWP_NS"

path_nosphere = r"C:\data\20171107\laser_noise_no_sphere"


file_list_l = glob.glob(path_l+"\*.h5")
file_list_h = glob.glob(path_h+r"\1mbar_zcool_G5_att.h5")
file_list_nosphere = glob.glob(path_nosphere+r"\dc_-400_1375mA.h5")


NFFT = 2**13

startfile = 500
endfile = 600

def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist

file_list_l = list_file_time_order(file_list_l)

file_list_l = file_list_l[startfile:endfile]

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
        before_chan, freqs = matplotlib.mlab.psd(dat[:, bu.xl]-numpy.mean(dat[:, bu.xl]), Fs = Fs, NFFT = NFFT)
        aux = np.argmax(before_chan[3:-1])
        freq_drive = freqs[aux]
	return [freqs, xpsd, before_chan, freq_drive, aux]

def unite_psd(file_list): # only for the signal before chamber
    freqs = np.array(getdata(file_list[0])[0])
    before = np.zeros(len(freqs))
    for file in file_list:
       a = getdata(file)
       before += np.array(a[2])
    return [freqs, before/len(file_list)]

L = unite_psd(file_list_l)

H = getdata(file_list_h[0])

N = getdata(file_list_nosphere[0])



plt.figure()
plt.loglog(H[0],np.sqrt(H[2]))
plt.loglog(L[0],np.sqrt(L[1]))
plt.loglog(N[0],np.sqrt(N[2]))
plt.show()
