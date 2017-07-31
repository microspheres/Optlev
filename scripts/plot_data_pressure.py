import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob

refname = r"1mbar_zcool_G5_att.h5"
fname0 = r""
path_charge = r"C:\data\20170726\bead4_15um_QWP\calibration_47_3Hz"

filelist_charge = glob.glob(path_charge+"\*.h5")
		 

Fs = 10e3  ## this is ignored with HDF5 files
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
	ypsd, freqs = matplotlib.mlab.psd(dat[:, bu.yi]-numpy.mean(dat[:, bu.yi]), Fs = Fs, NFFT = NFFT)
        zpsd, freqs = matplotlib.mlab.psd(dat[:, bu.zi]-numpy.mean(dat[:, bu.zi]), Fs = Fs, NFFT = NFFT)
        xpsd_old, freqs = matplotlib.mlab.psd(dat[:, bu.xi_old]-numpy.mean(dat[:, bu.xi_old]), Fs = Fs, NFFT = NFFT)
        drive, freqs = matplotlib.mlab.psd(dat[:, bu.drive]-numpy.mean(dat[:, bu.drive]), Fs = Fs, NFFT = NFFT) 

	return [xpsd, drive]

def calibration_1e(filelist):
        psdx = np.zeros(len(getdata(filelist[0])[0]))
        b1 = np.zeros(len(filelist))
        b2 = np.zeros(len(filelist))
        for i in range(len(filelist)):
                b1[i], b2[i] = getdata(filelist[i])
                drive = np.argmax(b2[i])
                a = np.argmax(drive)
                psdx += b2[i]
        psdx = np.sqrt(psdx/len(filelist))
        return [psdx, drive]

psdx, d = calibration_1e(filelist_charge)
