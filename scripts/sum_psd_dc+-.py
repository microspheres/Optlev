import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import os, re, time, glob


startfile = 0
endfile = 200

path = r"C:\data\20170925\bead4_15um_QWP_NS\steps\DC"

file_list = glob.glob(path+"\*.h5")

def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist

file_list = list_file_time_order(file_list)

file_list = file_list[startfile:endfile]


def get_specific_DCp(list):
    file_list_new = []
    for i in range(len(list)):
        if float(re.findall("-?\d+mVDC",list[i])[0][:-4]) == 18800:
            file_list_new.append(list[i])
    return file_list_new

def get_specific_DCm(list):
    file_list_new = []
    for i in range(len(list)):
        if float(re.findall("-?\d+mVDC",list[i])[0][:-4]) == -18800:
            file_list_new.append(list[i])
    return file_list_new

file_listp = get_specific_DCp(file_list)
file_listm = get_specific_DCm(file_list)

l1 = len(file_listp)
l2 = len(file_listm)
lmin = np.min([l1,l2])

file_listp = file_listp[:lmin]
file_listm = file_listm[:lmin]

		 

Fs = 10e3  ## this is ignored with HDF5 files
NFFT = 2**19

def get_x(fname):
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

	x = dat[:, bu.xi]-numpy.mean(dat[:, bu.xi])
	
	return x


def sum_time_stream(file_list):
    xs = 0
    for i in range(len(file_list)):
        x = get_x(file_list[i])
        xs = xs + x
    return xs

xsp = sum_time_stream(file_listp)
xsm = sum_time_stream(file_listm)

xs1 = xsp + xsm
xs2 = xsp + xsp
xpsd1, f1 = matplotlib.mlab.psd(xs1, Fs = Fs, NFFT = NFFT)
xpsd2, f2 = matplotlib.mlab.psd(xs2, Fs = Fs, NFFT = NFFT)

plt.figure
plt.loglog(f1,xpsd1, label = "oposite DC = 18800")
plt.loglog(f2,xpsd2, label = "same DC = 18800")
plt.legend()
plt.show()






