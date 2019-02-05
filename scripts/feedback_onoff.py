import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
import scipy.optimize as opt

path_list = [r"C:\data\20190202\15um\2\PID\comx7"]


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
                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )

	x = dat[:, bu.xi]-numpy.mean(dat[:, bu.xi])
        trigger = dat[:, 4]

	return [x, trigger, PID]

    
def get_files_path(path):
    file_list = glob.glob(path+"\*.h5")
    return file_list

def get_data_path(path):
    A = []
    for i in get_files_path(path):
        a = getdata(i)
        A.append(a)
    return A
        
plt.figure()
plt.plot(get_data_path(path_list[0])[0][1])
plt.plot(get_data_path(path_list[0])[0][0])
plt.show()

