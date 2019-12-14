import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob

refname = r"open.h5"
fname0 = r"closed.h5"
refname1 = r"lp.h5"
path = r"C:\data\20190912\prechamber\qfd\1"

make_plot_vs_time = True
conv_fac = 4.4e-14
if fname0 == "":
	filelist = glob.glob(path+"\*.h5")

	mtime = 0
	mrf = ""
	for fin in filelist:
		f = os.path.join(path, fin) 
		if os.path.getmtime(f)>mtime:
			mrf = f
			mtime = os.path.getmtime(f) 
 
	fname0 = mrf		


		 

Fs = 10e3  ## this is ignored with HDF5 files
NFFT = 2**13

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
                pid = dset.attrs['PID']
		
		#dat = 1.0*dat*max_volt/nbit
                dat = dat * 10./(2**15 - 1)
                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )



        x = 1.0*dat[:, 4]/dat[:, 6]
        y = 1.0*dat[:, 5]/dat[:, 6]

        
	xpsd, freqs = matplotlib.mlab.psd(x - numpy.mean(x), Fs = Fs, NFFT = NFFT) 
	ypsd, freqs = matplotlib.mlab.psd(y - numpy.mean(y), Fs = Fs, NFFT = NFFT)
        zpsd, freqs = matplotlib.mlab.psd(dat[:, 6]-numpy.mean(dat[:, 6]), Fs = Fs, NFFT = NFFT)

	return [freqs, xpsd, ypsd, dat, zpsd]

data0 = getdata(os.path.join(path, fname0))




if refname:
	data1 = getdata(os.path.join(path, refname))
        data2 = getdata(os.path.join(path, refname1))
Fs = 10000
b, a = sp.butter(1, [2*5./Fs, 2*10./Fs], btype = 'bandpass')

if make_plot_vs_time:	

        fig = plt.figure()
        plt.subplot(3, 1, 1)

        plt.plot(data0[3][:,4] - np.mean(data0[3][:, 4]) )
        if(refname):
                plt.plot(data1[3][:, 4] - np.mean(data1[3][:, 4]) )

        plt.subplot(3, 1, 2)
        plt.plot(data0[3][:, 5] - np.mean(data0[3][:, 5]) )
        if(refname):
                plt.plot(data1[3][:, 5] - np.mean(data1[3][:, 5]) )

        plt.subplot(3, 1, 3)
        plt.plot(data0[3][:, 6] - np.mean(data0[3][:, 6]) )
        if(refname):
                plt.plot(data1[3][:, 6] - np.mean(data1[3][:, 6]) )
       

fig = plt.figure()
plt.subplot(3, 1, 1)
plt.loglog(data0[0], np.sqrt(data0[1]),label="Closed")
if refname:
	plt.loglog(data1[0], np.sqrt(data1[1]),label="Open")
        plt.loglog(data2[0], np.sqrt(data2[1]),label="Low_press")
plt.ylabel("V/rtHz")
plt.legend(loc=1)
plt.subplot(3, 1, 2)
plt.loglog(data0[0], np.sqrt(data0[2]))
if refname:
	plt.loglog(data1[0], np.sqrt(data1[2]))
        plt.loglog(data2[0], np.sqrt(data2[2]))
plt.subplot(3, 1, 3)
plt.loglog(data0[0],  np.sqrt(data0[4]))
if refname:
	plt.loglog(data1[0], np.sqrt(data1[4]))
        plt.loglog(data2[0], np.sqrt(data2[4]))
plt.ylabel("V/rtHz")
plt.xlabel("Frequency[Hz]")
plt.show()

# fig = plt.figure
# plt.plot(data0[0], np.sqrt(data0[1]))
# plt.show()
