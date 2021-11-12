import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
import scipy.optimize as opt

# folder_HP = r"C:\data\20191022\10um\prechamber_LP\1\2mbar"
# xyz = r"2mbar_xyzcool.h5"
# yz = r"2mbar_yzcool.h5"

# NFFT = 2**15

def getdata(fname, NFFT):
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
                PID = dset.attrs['PID']
		
		#dat = 1.0*dat*max_volt/nbit
                dat = dat * 10./(2**15 - 1)
                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )

	xpsd, freqs = matplotlib.mlab.psd(dat[:, bu.xi]-numpy.mean(dat[:, bu.xi]), Fs = Fs, NFFT = NFFT)

	return [freqs, xpsd, np.abs(PID[0])]

def return_xpsd_xyzcool_yz_cool(folder, xyz, yz, NFFT):
    xyz_file = os.path.join(folder, xyz)
    yz_file = os.path.join(folder, yz)

    XYZ = getdata(xyz_file, NFFT)
    YZ = getdata(yz_file, NFFT)

    return [XYZ, YZ]


if __name__ == "__main__":
    return_xpsd_xyzcool_yz_cool(folder_HP, xyz, yz, NFFT)
