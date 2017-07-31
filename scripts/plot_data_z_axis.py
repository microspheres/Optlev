import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob


path = r"C:\data\ACDC_test\ACDC_balanced"
# refname = r"C:\data\20170403\bead6_15um"
# fname0 = r"xout_100Hz_1.h5"
# path = r"C:\Data\20170224\xy_test\feedback_test"

conv_fac = 4.4e-14
		 

Fs = 10e3  ## this is ignored with HDF5 files
NFFT = 2**19

def getdata_z(fname):
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


        z = numpy.mean(dat[:, bu.drive])
        return z

def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist

file_list = glob.glob(path+"\*.h5")



def z_position(file_list):
        z = []
        for i in range(len(file_list)):
                a = getdata_z(file_list[i])
                z.append(a)
        return z

z = z_position(list_file_time_order(file_list))

print z
print 'voltage after trek'
print np.mean(z)*200.0

plt.figure()
plt.plot(z)
plt.show()
        


